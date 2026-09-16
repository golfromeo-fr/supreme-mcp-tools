#!/usr/bin/env bash
# publish-image.sh — build the versioned node image and (optionally) push it
# to a registry. Local versioned builds work with no credentials; pushing
# needs a registry + `podman login` done beforehand.
#
# Usage:
#   ./deploy/publish-image.sh                 # build + tag locally (no push)
#   ./deploy/publish-image.sh ghcr.io/golfromeo-fr   # build + tag + PUSH
#   REGISTRY=... ./deploy/publish-image.sh    # env form
#
# Tags produced (VERSION = git describe --tags --always, e.g. m4-14-ga42492b):
#   <registry>/mcp-node:<version>   exact build
#   <registry>/mcp-node:latest      moving pointer
# With no REGISTRY: localhost/mcp-node:<version> + localhost/mcp-node:latest.
set -euo pipefail
cd "$(dirname "$0")/.."

REGISTRY="${REGISTRY:-${1:-}}"
VERSION="$(git describe --tags --always --dirty)"
SHA="$(git rev-parse HEAD)"

echo "[publish] building mcp-node ${VERSION} (${SHA:0:10})"
podman build \
  --build-arg IMAGE_VERSION="${VERSION}" \
  --build-arg GIT_SHA="${SHA}" \
  -f deploy/Containerfile \
  -t mcp-node:latest \
  -t "mcp-node:${VERSION}" \
  .

if [ -n "${REGISTRY}" ]; then
  for tag in latest "${VERSION}"; do
    podman tag "mcp-node:${tag}" "${REGISTRY}/mcp-node:${tag}"
    podman push "${REGISTRY}/mcp-node:${tag}"
    echo "[publish] pushed ${REGISTRY}/mcp-node:${tag}"
  done
  echo "[publish] image labels:"
  podman inspect "mcp-node:${VERSION}" --format \
    '{{index .Labels "org.opencontainers.image.version"}} / {{index .Labels "org.opencontainers.image.revision"}}'
else
  echo "[publish] local-only build (pass a REGISTRY arg to push)"
fi
