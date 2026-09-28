"""Deployment-config regression guards (review plan L1, plus positive
guards for the C1/H1/H2 security band).

L1 origin: the OCR review suspected the nginx LB echoed request
Authorization headers back to clients, then refuted itself — add_header
adds a RESPONSE header and never reflects request headers. These asserts
keep it that way: any future nginx-lb.conf edit that introduces request-
header reflection ($http_*, Authorization) into a response header fails
here. The positive guards pin the C1 (loopback bind), H1 (ip_hash), and
H2 (SSE/streaming proxy settings) fixes so they can't silently regress.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_LB_CONF = PROJECT_ROOT / "deploy" / "nginx-lb.conf"
_LB_COMPOSE = PROJECT_ROOT / "deploy" / "compose-lb.yml"
_CONTAINERFILE = PROJECT_ROOT / "deploy" / "Containerfile"


def test_nginx_lb_never_reflects_request_headers():
    """No add_header may echo request headers (Authorization et al)."""
    for line in _LB_CONF.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("add_header"):
            low = stripped.lower()
            assert "authorization" not in low and "$http_" not in low, (
                f"response header reflects a request header: {stripped!r}")


def test_nginx_lb_keeps_sticky_sessions_and_streaming():
    """H1: ip_hash must stay on the upstream (stateful /mcp sessions are
    node-local). H2: SSE/long-poll needs HTTP/1.1 + no buffering + long
    timeouts (nginx's 60s default kills them mid-stream)."""
    conf = _LB_CONF.read_text()
    assert "ip_hash;" in conf
    assert "proxy_http_version 1.1;" in conf
    assert "proxy_buffering off;" in conf
    assert "proxy_read_timeout" in conf


def test_lb_published_on_loopback_only():
    """C1: the LB fronts the CENTRAL management API (users/roles/env
    mutation) — it must not bind every interface."""
    assert '"127.0.0.1:18080:80"' in _LB_COMPOSE.read_text()


def test_containerfile_never_bakes_real_tool_api_keys():
    """P0 (m4 review): tools/*/config.json carry real auth.api_key values and
    publish-image.sh can push the image. The build must sanitize them to the
    "image-sanitized" placeholder in a THROWAWAY stage and copy only the
    sanitized tree into the final image, so no real key reaches any image
    layer — while the non-secret blocks (ragmcp embedding, convertermcp
    allowed_roots) stay intact."""
    cf = _CONTAINERFILE.read_text()
    assert "image-sanitized" in cf, "tool auth.api_key sanitize step missing"
    assert "AS tools-sanitized" in cf, "sanitize must run in a throwaway stage"
    assert "COPY --from=tools-sanitized" in cf, (
        "final image must take tools/ from the sanitized stage, not a raw COPY")
