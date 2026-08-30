#!/usr/bin/env python3
"""
WebMCP Server - FastMCP Implementation
"""
import sys
import os
import logging
import time
import asyncio
import re
import hashlib
import threading
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlencode, urlparse
import httpx

TOOL_NAME = "webmcp"

try:
    from launcher.launcher_config import load_ports_config
    ports_config = load_ports_config()
    MCP_PORT = int(os.environ.get("MCP_PORT", ports_config["assignments"]["mcp"][TOOL_NAME]))
    MGMT_PORT = int(os.environ.get("MCP_MGMT_PORT", ports_config["assignments"]["mgmt"][TOOL_NAME]))
except Exception as e:
    print(f"ERROR: Failed to load ports.json: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(TOOL_NAME)

BRAVE_SEARCH_API_KEY = os.environ.get("BRAVE_SEARCH_API_KEY", "")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY", "")
logger.info(f"BRAVE_SEARCH_API_KEY loaded: {'SET' if BRAVE_SEARCH_API_KEY else 'NOT SET'}")
logger.info(f"SERPAPI_API_KEY loaded: {'SET' if SERPAPI_API_KEY else 'NOT SET'}")

try:
    from tools.fef_integration import ToolExtensionManager, register_common_extensions, setup_tool_extensions
    from launcher.tool_extensions import Extension, ExtensionType
    FEF_V3_AVAILABLE = True
    logger.info("FEF V3 modules loaded successfully")
except ImportError as e:
    FEF_V3_AVAILABLE = False
    logger.warning(f"FEF V3 not available: {e}")

webmcp_metrics = {
    "search_count": 0,
    "fetch_count": 0,
    "search_errors": 0,
    "fetch_errors": 0,
    "total_search_time_ms": 0.0,
    "total_fetch_time_ms": 0.0,
    "min_search_time_ms": None,
    "max_search_time_ms": 0.0,
    "min_fetch_time_ms": float('inf'),
    "max_fetch_time_ms": 0.0,
}

def get_search_stats(params: dict[str, Any]) -> dict[str, Any]:
    avg_search_time = webmcp_metrics["total_search_time_ms"] / webmcp_metrics["search_count"] if webmcp_metrics["search_count"] > 0 else 0.0
    return {"total_searches": webmcp_metrics["search_count"], "search_errors": webmcp_metrics["search_errors"], "avg_search_time_ms": round(avg_search_time, 2)}

def get_fetch_stats(params: dict[str, Any]) -> dict[str, Any]:
    avg_fetch_time = webmcp_metrics["total_fetch_time_ms"] / webmcp_metrics["fetch_count"] if webmcp_metrics["fetch_count"] > 0 else 0.0
    return {"total_fetches": webmcp_metrics["fetch_count"], "fetch_errors": webmcp_metrics["fetch_errors"], "avg_fetch_time_ms": round(avg_fetch_time, 2)}

def get_search_history(params: dict[str, Any]) -> dict[str, Any]:
    limit = params.get("limit", 10)
    return {"recent_searches": webmcp_metrics.get("recent_searches", [])[-limit:], "total": len(webmcp_metrics.get("recent_searches", []))}

def get_fetch_cache_hits(params: dict[str, Any]) -> dict[str, Any]:
    cache_hits = webmcp_metrics.get("cache_hits", 0)
    cache_misses = webmcp_metrics.get("cache_misses", 0)
    total = cache_hits + cache_misses
    hit_ratio = cache_hits / total if total > 0 else 0.0
    return {"cache_hits": cache_hits, "cache_misses": cache_misses, "hit_ratio": round(hit_ratio, 3)}

from tools.shared.utils import is_internal_url as _is_internal_url

# KNOWN LIMITATION: DNS rebinding / TOCTOU — DNS is resolved twice (once here,
# once by httpx). follow_redirects is limited to max_redirects=5 and the final
# URL is checked after redirects, but per-redirect validation requires a custom
# transport which is not yet implemented.

class SimpleCache:
    MAX_SIZE = 1000

    def __init__(self, default_ttl: int = 3600):
        self.cache: dict[str, tuple[str, float]] = {}
        self.lock = threading.RLock()
        self.default_ttl = default_ttl

    def get(self, key: str) -> str | None:
        with self.lock:
            if key in self.cache:
                content, expiry = self.cache[key]
                if time.time() < expiry:
                    return content
                else:
                    del self.cache[key]
        return None

    def set(self, key: str, value: str, ttl: int | None = None) -> None:
        with self.lock:
            if len(self.cache) >= self.MAX_SIZE:
                expired_keys = [k for k, (_, exp) in self.cache.items() if time.time() >= exp]
                if expired_keys:
                    for k in expired_keys:
                        del self.cache[k]
                else:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
            expiry = time.time() + (ttl if ttl is not None else self.default_ttl)
            self.cache[key] = (value, expiry)
    
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
    
    def cleanup_expired(self) -> None:
        with self.lock:
            current_time = time.time()
            expired_keys = [key for key, (_, expiry) in self.cache.items() if current_time >= expiry]
            for key in expired_keys:
                del self.cache[key]

_cache = SimpleCache(default_ttl=3600)

def _generate_cache_key(url: str, params: dict) -> str:
    try:
        param_str = str(sorted(params.items(), key=lambda x: str(x[0])))
    except TypeError:
        return "error:mixed-type-keys"
    combined = f"{url}:{param_str}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]
def _clean_html(html_content: str, include_images: bool = True, include_tables: bool = True, include_links: bool = True) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return html_content
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup(['script', 'style', 'noscript', 'iframe']):
        tag.decompose()
    for selector in ['nav', 'header', 'footer', '[role="navigation"]', '[role="banner"]', '[role="contentinfo"]', '.navigation', '.nav', '.menu', '.sidebar', '.footer', '.header', '.cookie-banner', '.advertisement', '.ads', '.promo']:
        for element in soup.select(selector):
            element.decompose()
    if not include_images:
        for img in soup.find_all('img'):
            img.decompose()
    if not include_tables:
        for table in soup.find_all('table'):
            table.decompose()
    if not include_links:
        for a in soup.find_all('a'):
            a.replace_with(a.get_text())
    return str(soup)

def _html_to_markdown(html_content: str) -> str:
    try:
        from bs4 import BeautifulSoup, NavigableString
    except ImportError:
        return html_content
    soup = BeautifulSoup(html_content, 'html.parser')
    markdown = []
    _visited = set()
    def _render(el, list_depth=0):
        if el in _visited:
            return
        _visited.add(el)
        if isinstance(el, NavigableString):
            text = str(el).strip()
            if text:
                markdown.append(text)
            return
        if el.name in ('script', 'style', 'noscript', 'iframe'):
            return
        if el.name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            level = int(el.name[1])
            markdown.append(f"\n{'#' * level} {el.get_text().strip()}\n")
            return
        if el.name == 'p':
            markdown.append(f"\n{el.get_text().strip()}\n")
            return
        if el.name in ('strong', 'b'):
            markdown.append(f"**{el.get_text()}**")
            return
        if el.name in ('em', 'i'):
            markdown.append(f"*{el.get_text()}*")
            return
        if el.name == 'a':
            href = el.get('href', '')
            text = el.get_text()
            if href and text:
                markdown.append(f"[{text}]({href})")
            return
        if el.name == 'li':
            prefix = '  ' * list_depth + '- '
            markdown.append(f"{prefix}{el.get_text().strip()}")
            for child in el.children:
                if hasattr(child, 'name') and child.name not in (None, 'ul', 'ol'):
                    _render(child)
            return
        if el.name in ('ul', 'ol'):
            markdown.append("\n")
            for child in el.children:
                if hasattr(child, 'name'):
                    _render(child, list_depth + 1)
            return
        if el.name == 'br':
            markdown.append("\n")
            return
        if el.name == 'hr':
            markdown.append("\n---\n")
            return
        for child in el.children:
            _render(child, list_depth)
    _render(soup)
    return ''.join(markdown)

def _extract_text_content(html_content: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return html_content
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator='\n', strip=True)
# ============================================================================
# FastMCP Instance (via shared factory — DualHeaderVerifier auth)
# ============================================================================

from tools.shared.server_factory import create_fastmcp_server, DEFAULT_HOST

mcp = create_fastmcp_server(TOOL_NAME)


@mcp.tool()
async def brave_search_web(query: str, count: int = 10, timeout: float = 30.0, language: str = "en") -> str:
    """Handler for Brave web search with enhanced features (web scraping)"""
    if not query:
        return "Error: 'query' parameter is required"

    start_time = time.perf_counter()

    try:
        params = {'q': query, 'count': str(count), 'source': 'web', 'lang': language}
        search_url = f"https://search.brave.com/search?{urlencode(params)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": f"{language},en;q=0.9",
            "Connection": "keep-alive",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
        }

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
            response = await client.get(search_url)
            response.raise_for_status()
            content = response.text
            if len(content) > 100000:
                content = content[:100000] + "\n\n[Content truncated due to size limit]"
            metadata = f"## Search Results Metadata\n- Query: {query}\n- Results Count: {count}\n- Language: {language}\n- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            formatted_content = metadata + content
            logger.info(f"Enhanced Brave search completed for query: {query}")

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            webmcp_metrics["search_count"] += 1
            webmcp_metrics["total_search_time_ms"] += elapsed_ms
            if webmcp_metrics["min_search_time_ms"] is None or elapsed_ms < webmcp_metrics["min_search_time_ms"]:
                webmcp_metrics["min_search_time_ms"] = elapsed_ms
            if elapsed_ms > webmcp_metrics["max_search_time_ms"]:
                webmcp_metrics["max_search_time_ms"] = elapsed_ms
            if fef_manager is not None:
                fef_manager.metrics.record_request(
                    endpoint="tools/call", tool_name="brave_search_web",
                    success=True, duration_ms=elapsed_ms
                )

            return formatted_content
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code} error: {e.response.reason_phrase}"
        logger.error(error_msg)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        webmcp_metrics["search_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="brave_search_web",
                success=False, duration_ms=elapsed_ms
            )

        return f"Error: {error_msg}"
    except httpx.RequestError as e:
        error_msg = f"Network error: {str(e)}"
        logger.error(error_msg)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        webmcp_metrics["search_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="brave_search_web",
                success=False, duration_ms=elapsed_ms
            )

        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        webmcp_metrics["search_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="brave_search_web",
                success=False, duration_ms=elapsed_ms
            )

        return f"Error: {error_msg}"
@mcp.tool()
async def brave_search_api(query: str, count: int = 10, timeout: float = 30.0, language: str = "en", country: str = "US", text_decorations: bool = True, fresh: bool = False, use_post: bool = False, llm_mode: bool = False) -> str:
    """
    Handler for Brave Search API with enhanced features.
    Supports both GET and POST methods to the /res/v1/web/search endpoint.
    """
    logger.info(f"brave_search_api called - BRAVE_SEARCH_API_KEY: {'SET' if BRAVE_SEARCH_API_KEY else 'NOT SET'}")

    if not query:
        return "Error: 'query' parameter is required"

    if not BRAVE_SEARCH_API_KEY:
        logger.warning("brave_search_api called but BRAVE_SEARCH_API_KEY is not set in .env")
        webmcp_metrics["search_errors"] += 1
        webmcp_metrics["search_count"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="brave_search_api",
                success=False, duration_ms=0.0
            )
        return (
            "BRAVE_SEARCH_API_KEY is not configured.\n\n"
            "To use brave_search_api, you need to:\n"
            "1. Get a Brave Search API key from: https://brave.com/search/api/\n"
            "2. Add it to your .env file: BRAVE_SEARCH_API_KEY=your_api_key_here\n\n"
            "Alternative tools that don't require an API key:\n"
            "- brave_search_web: Web scraping-based search (free, no API key needed)\n"
            "- fetch_url: Fetch and process any URL content directly"
        )

    start_time = time.perf_counter()

    try:
        import json
        base_url = "https://api.search.brave.com/res/v1/web/search"
        headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_SEARCH_API_KEY}

        async with httpx.AsyncClient(timeout=timeout) as client:
            if use_post:
                headers["Content-Type"] = "application/json"
                body = {"q": query, "count": min(max(count, 1), 20), "country": country, "search_lang": language, "text_decorations": text_decorations, "fresh": fresh}
                response = await client.post(base_url, headers=headers, json=body)
            else:
                params = {"q": query, "count": min(max(count, 1), 20), "country": country, "search_lang": language, "text_decorations": "true" if text_decorations else "false", "fresh": "true" if fresh else "false"}
                response = await client.get(base_url, headers=headers, params=params)

            response.raise_for_status()
            result_data = response.json()

            if llm_mode:
                llm_results = {"query": query, "results": []}
                if "web" in result_data and "results" in result_data["web"]:
                    for result in result_data["web"]["results"]:
                        llm_results["results"].append({"title": result.get("title", ""), "url": result.get("url", ""), "description": result.get("description", ""), "type": "web"})
                if "news" in result_data and "results" in result_data["news"]:
                    for result in result_data["news"]["results"]:
                        llm_results["results"].append({"title": result.get("title", ""), "url": result.get("url", ""), "description": result.get("description", ""), "age": result.get("age", ""), "type": "news"})
                logger.info(f"Brave Search API completed for query: {query} (llm_mode)")

                elapsed_ms = (time.perf_counter() - start_time) * 1000
                webmcp_metrics["search_count"] += 1
                webmcp_metrics["total_search_time_ms"] += elapsed_ms
                if webmcp_metrics["min_search_time_ms"] is None or elapsed_ms < webmcp_metrics["min_search_time_ms"]:
                    webmcp_metrics["min_search_time_ms"] = elapsed_ms
                if elapsed_ms > webmcp_metrics["max_search_time_ms"]:
                    webmcp_metrics["max_search_time_ms"] = elapsed_ms
                if fef_manager is not None:
                    fef_manager.metrics.record_request(
                        endpoint="tools/call", tool_name="brave_search_api",
                        success=True, duration_ms=elapsed_ms
                    )

                return json.dumps(llm_results, indent=2)

            formatted_results = []
            formatted_results.append(f"## Brave Search API Results\n")
            formatted_results.append(f"- Query: {query}\n")
            formatted_results.append(f"- Results Count: {count}\n")
            formatted_results.append(f"- Language: {language}\n")
            formatted_results.append(f"- Country: {country}\n")
            formatted_results.append(f"- Method: {'POST' if use_post else 'GET'}\n")
            formatted_results.append(f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if "query" in result_data:
                query_info = result_data["query"]
                formatted_results.append("### Query Information\n\n")
                if "original" in query_info:
                    formatted_results.append(f"- Original Query: {query_info['original']}\n")
                if "cleaned" in query_info:
                    formatted_results.append(f"- Cleaned Query: {query_info['cleaned']}\n")
                if "altered" in query_info:
                    formatted_results.append(f"- Altered Query: {query_info['altered']}\n")
                formatted_results.append("\n")

            if "web" in result_data and "results" in result_data["web"]:
                formatted_results.append("### Web Results\n\n")
                for i, result in enumerate(result_data["web"]["results"], 1):
                    formatted_results.append(f"{i}. **{result.get('title', 'No title')}**\n")
                    formatted_results.append(f"   URL: {result.get('url', 'No URL')}\n")
                    formatted_results.append(f"   Description: {result.get('description', 'No description')}\n\n")

            logger.info(f"Brave Search API completed for query: {query} (method: {'POST' if use_post else 'GET'})")

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            webmcp_metrics["search_count"] += 1
            webmcp_metrics["total_search_time_ms"] += elapsed_ms
            if webmcp_metrics["min_search_time_ms"] is None or elapsed_ms < webmcp_metrics["min_search_time_ms"]:
                webmcp_metrics["min_search_time_ms"] = elapsed_ms
            if elapsed_ms > webmcp_metrics["max_search_time_ms"]:
                webmcp_metrics["max_search_time_ms"] = elapsed_ms
            if fef_manager is not None:
                fef_manager.metrics.record_request(
                    endpoint="tools/call", tool_name="brave_search_api",
                    success=True, duration_ms=elapsed_ms
                )

            return ''.join(formatted_results)
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code} error: {e.response.reason_phrase}"
        logger.error(error_msg)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        webmcp_metrics["search_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="brave_search_api",
                success=False, duration_ms=elapsed_ms
            )

        return f"Error: {error_msg}"
    except httpx.RequestError as e:
        error_msg = f"Network error: {str(e)}"
        logger.error(error_msg)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        webmcp_metrics["search_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="brave_search_api",
                success=False, duration_ms=elapsed_ms
            )

        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        webmcp_metrics["search_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="brave_search_api",
                success=False, duration_ms=elapsed_ms
            )

        return f"Error: {error_msg}"
@mcp.tool()
async def google_search_api(query: str, engine: str = "google", google_domain: str = "google.com", hl: str = "en", gl: str = "us", location: str | None = None, start: int = 0, num: int = 10, safe: str = "active", device: str | None = None, llm_mode: bool = False) -> str:
    """
    Handler for Google Search API using SerpAPI.
    """
    logger.info(f"google_search_api called - SERPAPI_API_KEY: {'SET' if SERPAPI_API_KEY else 'NOT SET'}")

    if not query:
        return "Error: 'query' parameter is required"

    if not SERPAPI_API_KEY:
        logger.warning("google_search_api called but SERPAPI_API_KEY is not set in .env")
        webmcp_metrics["search_errors"] += 1
        webmcp_metrics["search_count"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="google_search_api",
                success=False, duration_ms=0.0
            )
        return (
            "SERPAPI_API_KEY is not configured.\n\n"
            "To use google_search_api, you need to:\n"
            "1. Get a SerpAPI key from: https://serpapi.com/\n"
            "2. Add it to your .env file: SERPAPI_API_KEY=your_api_key_here\n\n"
            "Alternative tools that don't require an API key:\n"
            "- brave_search_web: Web scraping-based search (free, no API key needed)\n"
            "- fetch_url: Fetch and process any URL content directly"
        )

    start_time = time.perf_counter()

    try:
        import json
        from serpapi import GoogleSearch
        params = {"engine": engine, "q": query, "google_domain": google_domain, "hl": hl, "gl": gl, "api_key": SERPAPI_API_KEY, "num": min(max(num, 1), 100)}
        if location:
            params["location"] = location
        if start > 0:
            params["start"] = str(start)
        if safe:
            params["safe"] = safe
        if device and device.strip():
            params["device"] = device

        search = GoogleSearch(params)
        results = await asyncio.to_thread(search.get_dict)
        logger.info(f"Raw SerpAPI response keys: {list(results.keys())}")

        if llm_mode:
            llm_results = {"query": query, "results": []}
            if "organic_results" in results:
                for result in results["organic_results"]:
                    llm_results["results"].append({"title": result.get("title", ""), "url": result.get("link", ""), "snippet": result.get("snippet", "")})
            if "answer_box" in results:
                ab = results["answer_box"]
                llm_results["answer"] = ab.get("answer", ab.get("snippet", ""))
            if "knowledge_graph" in results:
                kg = results["knowledge_graph"]
                llm_results["knowledge_graph"] = {"title": kg.get("title", ""), "type": kg.get("type", ""), "description": kg.get("description", ""), "website": kg.get("website", "")}
            logger.info(f"Google Search API (SerpAPI) completed for query: {query} (llm_mode)")

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            webmcp_metrics["search_count"] += 1
            webmcp_metrics["total_search_time_ms"] += elapsed_ms
            if webmcp_metrics["min_search_time_ms"] is None or elapsed_ms < webmcp_metrics["min_search_time_ms"]:
                webmcp_metrics["min_search_time_ms"] = elapsed_ms
            if elapsed_ms > webmcp_metrics["max_search_time_ms"]:
                webmcp_metrics["max_search_time_ms"] = elapsed_ms
            if fef_manager is not None:
                fef_manager.metrics.record_request(
                    endpoint="tools/call", tool_name="google_search_api",
                    success=True, duration_ms=elapsed_ms
                )

            return json.dumps(llm_results, indent=2)

        formatted_results = []
        formatted_results.append(f"## Google Search API Results (SerpAPI)\n")
        formatted_results.append(f"- Query: {query}\n")
        formatted_results.append(f"- Engine: {engine}\n")
        formatted_results.append(f"- Google Domain: {google_domain}\n")
        formatted_results.append(f"- Language: {hl}\n")
        formatted_results.append(f"- Country: {gl}\n")
        if location:
            formatted_results.append(f"- Location: {location}\n")
        formatted_results.append(f"- Results Count: {num}\n")
        if start > 0:
            formatted_results.append(f"- Start Offset: {start}\n")
        formatted_results.append(f"- Safe Search: {safe}\n")
        if device:
            formatted_results.append(f"- Device: {device}\n")
        formatted_results.append(f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if "error" in results:
            error_info = results["error"]
            formatted_results.append("### Error\n\n")
            if isinstance(error_info, dict):
                formatted_results.append(f"Error Code: {error_info.get('code', 'Unknown')}\n")
                formatted_results.append(f"Error Message: {error_info.get('message', 'No message')}\n")
            else:
                formatted_results.append(f"Error: {str(error_info)}\n")
            logger.error(f"SerpAPI error: {error_info}")

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            webmcp_metrics["search_errors"] += 1
            if fef_manager is not None:
                fef_manager.metrics.record_request(
                    endpoint="tools/call", tool_name="google_search_api",
                    success=False, duration_ms=elapsed_ms
                )

            return ''.join(formatted_results)

        if "search_information" in results:
            search_info = results["search_information"]
            formatted_results.append("### Search Information\n\n")
            if "query_displayed" in search_info:
                formatted_results.append(f"- Query Displayed: {search_info['query_displayed']}\n")
            if "total_results" in search_info:
                formatted_results.append(f"- Total Results: {search_info['total_results']}\n")
            formatted_results.append("\n")

        if "answer_box" in results:
            answer_box = results["answer_box"]
            formatted_results.append("### Answer Box\n\n")
            if "title" in answer_box:
                formatted_results.append(f"**{answer_box['title']}**\n")
            if "answer" in answer_box:
                formatted_results.append(f"{answer_box['answer']}\n")
            elif "snippet" in answer_box:
                formatted_results.append(f"{answer_box['snippet']}\n")
            if "link" in answer_box:
                formatted_results.append(f"Source: {answer_box['link']}\n")
            formatted_results.append("\n")

        if "knowledge_graph" in results:
            kg = results["knowledge_graph"]
            formatted_results.append("### Knowledge Graph\n\n")
            if "title" in kg:
                formatted_results.append(f"**{kg['title']}**\n")
            if "type" in kg:
                formatted_results.append(f"Type: {kg['type']}\n")
            if "description" in kg:
                formatted_results.append(f"{kg['description']}\n")
            if "website" in kg:
                formatted_results.append(f"Website: {kg['website']}\n")
            formatted_results.append("\n")

        if "organic_results" in results:
            formatted_results.append("### Organic Results\n\n")
            for i, result in enumerate(results["organic_results"], 1):
                formatted_results.append(f"{i}. **{result.get('title', 'No title')}**\n")
                formatted_results.append(f"   URL: {result.get('link', 'No URL')}\n")
                if "snippet" in result:
                    formatted_results.append(f"   Description: {result['snippet']}\n")
                formatted_results.append("\n")

        if "related_searches" in results:
            formatted_results.append("### Related Searches\n\n")
            for i, related in enumerate(results["related_searches"], 1):
                formatted_results.append(f"{i}. {related.get('query', 'No query')}\n")
                formatted_results.append("\n")

        logger.info(f"Google Search API (SerpAPI) completed for query: {query}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        webmcp_metrics["search_count"] += 1
        webmcp_metrics["total_search_time_ms"] += elapsed_ms
        if webmcp_metrics["min_search_time_ms"] is None or elapsed_ms < webmcp_metrics["min_search_time_ms"]:
            webmcp_metrics["min_search_time_ms"] = elapsed_ms
        if elapsed_ms > webmcp_metrics["max_search_time_ms"]:
            webmcp_metrics["max_search_time_ms"] = elapsed_ms
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="google_search_api",
                success=True, duration_ms=elapsed_ms
            )

        return ''.join(formatted_results)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        webmcp_metrics["search_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="google_search_api",
                success=False, duration_ms=elapsed_ms
            )

        return f"Error: {error_msg}"
@mcp.tool()
async def post_url(url: str, data: str | None = None, headers: dict | None = None, timeout: float = 30.0) -> str:
    """Handler for sending POST requests to a URL with JSON payload support."""
    logger.debug(f"Processing post_url tool with url: {url}")

    if not url:
        logger.error("Missing required parameter: url")
        return "Missing required parameter: url"

    if _is_internal_url(url):
        return "Error: Internal URLs are not allowed for security reasons"

    headers = headers or {}

    start_time = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, max_redirects=5) as client:
            response = await client.post(url, data=data, headers=headers)
            if _is_internal_url(str(response.url)):
                return "Error: Redirected to internal URL"
            response.raise_for_status()
            logger.info(f"Successfully posted to URL: {url}")
            metadata_lines = ["## POST URL Results", f"- URL: {url}", f"- Status Code: {response.status_code}", f"- Content Type: {response.headers.get('content-type', 'unknown')}", f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}"]
            if data:
                metadata_lines.append(f"- Data: {data[:200] if len(data) > 200 else data}...")
            metadata = '\n'.join(metadata_lines) + '\n\n'

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            webmcp_metrics["fetch_count"] += 1
            webmcp_metrics["total_fetch_time_ms"] += elapsed_ms
            if elapsed_ms < webmcp_metrics["min_fetch_time_ms"]:
                webmcp_metrics["min_fetch_time_ms"] = elapsed_ms
            if elapsed_ms > webmcp_metrics["max_fetch_time_ms"]:
                webmcp_metrics["max_fetch_time_ms"] = elapsed_ms
            if fef_manager is not None:
                fef_manager.metrics.record_request(
                    endpoint="tools/call", tool_name="post_url",
                    success=True, duration_ms=elapsed_ms
                )

            return metadata + response.text
    except httpx.HTTPError as e:
        error_msg = f"HTTP error occurred while posting to URL {url}: {str(e)}"
        logger.error(error_msg)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        webmcp_metrics["fetch_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="post_url",
                success=False, duration_ms=elapsed_ms
            )

        return error_msg
    except Exception as e:
        error_msg = f"Error posting to URL {url}: {str(e)}"
        logger.error(error_msg)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        webmcp_metrics["fetch_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="post_url",
                success=False, duration_ms=elapsed_ms
            )

        return error_msg
@mcp.tool()
async def fetch_url(url: str, timeout: float = 30.0, max_length: int = 50000, start_index: int = 0, format: str = "markdown", follow_redirects: bool = True, headers: dict | None = None, response_type: str = "auto", include_images: bool = True, include_tables: bool = True, include_links: bool = True, use_cache: bool = False, cache_ttl: int = 3600, max_size: int = 10485760) -> str:
    """
    Enhanced URL fetch tool with comprehensive content processing.
    Based on best practices from mcp-server-fetch and html2md-mcp.
    """
    if not url:
        return "Error: 'url' parameter is required"

    if _is_internal_url(url):
        webmcp_metrics["fetch_errors"] += 1
        return "Error: Internal URLs are not allowed for security reasons"

    if timeout < 1 or timeout > 300:
        return "Error: 'timeout' must be between 1 and 300 seconds"
    if max_length < 100 or max_length > 1000000:
        return "Error: 'max_length' must be between 100 and 1000000 characters"
    if max_size < 1024 or max_size > 52428800:
        return "Error: 'max_size' must be between 1KB and 50MB"

    # Check cache if enabled
    if use_cache:
        cache_key = _generate_cache_key(url, {'format': format, 'include_images': include_images, 'include_tables': include_tables, 'include_links': include_links, 'start_index': start_index, 'max_length': max_length})
        cached_content = _cache.get(cache_key)
        if cached_content is not None:
            logger.info(f"Cache hit for URL: {url}")
            webmcp_metrics["cache_hits"] = webmcp_metrics.get("cache_hits", 0) + 1

            elapsed_ms = 0.0  # Cache hit, no actual fetch time
            webmcp_metrics["fetch_count"] += 1
            webmcp_metrics["total_fetch_time_ms"] += elapsed_ms
            if elapsed_ms < webmcp_metrics["min_fetch_time_ms"]:
                webmcp_metrics["min_fetch_time_ms"] = elapsed_ms
            if elapsed_ms > webmcp_metrics["max_fetch_time_ms"]:
                webmcp_metrics["max_fetch_time_ms"] = elapsed_ms
            if fef_manager is not None:
                fef_manager.metrics.record_request(
                    endpoint="tools/call", tool_name="fetch_url",
                    success=True, duration_ms=elapsed_ms
                )

            return f"## Web Reader Results (Cached)\n- URL: {url}\n- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n- Format: {format}\n\n{cached_content}"
        else:
            webmcp_metrics["cache_misses"] = webmcp_metrics.get("cache_misses", 0) + 1

    # No manual Accept-Encoding anywhere: httpx advertises only codecs it can
    # decode (gzip/deflate, plus br/zstd when those packages are installed).
    # A hand-set "br" without the brotli package made servers answer with
    # brotli and httpx returned the raw compressed bytes as text.
    default_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }

    if not use_cache:
        default_headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        default_headers["Pragma"] = "no-cache"

    merged_headers = {**default_headers, **(headers or {})}

    start_time = time.perf_counter()

    try:
        _cache.cleanup_expired()

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=follow_redirects, headers=merged_headers, max_redirects=5, limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)) as client:
            response = await client.get(url)
            if _is_internal_url(str(response.url)):
                webmcp_metrics["fetch_errors"] += 1
                return "Error: Redirected to internal URL"
            response.raise_for_status()

            content_size = len(response.content)
            if content_size > max_size:
                logger.warning(f"Content size {content_size} exceeds limit {max_size} for URL: {url}")

                elapsed_ms = (time.perf_counter() - start_time) * 1000
                webmcp_metrics["fetch_errors"] += 1
                if fef_manager is not None:
                    fef_manager.metrics.record_request(
                        endpoint="tools/call", tool_name="fetch_url",
                        success=False, duration_ms=elapsed_ms
                    )

                return f"Error: Content size ({content_size:,} bytes) exceeds maximum allowed size ({max_size:,} bytes)."

            content = response.text
            content_type = response.headers.get("content-type", "").lower()

            metadata_lines = ["## Web Reader Results", f"- URL: {url}", f"- Content Type: {content_type}", f"- Status Code: {response.status_code}", f"- Content Size: {content_size:,} bytes", f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", f"- Format: {format}", f"- Include Images: {include_images}", f"- Include Tables: {include_tables}", f"- Include Links: {include_links}", f"- Start Index: {start_index}", f"- Max Length: {max_length:,} characters"]
            metadata = '\n'.join(metadata_lines) + '\n\n'

            import json

            if response_type == "json" or (response_type == "auto" and "application/json" in content_type):
                try:
                    json_data = response.json()
                    content = json.dumps(json_data, indent=2)
                    if format == "markdown":
                        content = f"```json\n{content}\n```"
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON response from {url}")
                    content = response.text
                    if format == "markdown":
                        content = f"```\n{content}\n```"
            elif response_type == "html" or (response_type == "auto" and "text/html" in content_type):
                content = _clean_html(content, include_images=include_images, include_tables=include_tables, include_links=include_links)
                if format == "markdown":
                    content = _html_to_markdown(content)
                    content = f"# Content from {url}\n\n{content}"
                elif format == "text":
                    content = _extract_text_content(content)
            elif response_type == "text" or (response_type == "auto" and "text/plain" in content_type):
                if format == "markdown":
                    content = f"```\n{content}\n```"
            else:
                content = f"[Binary content - {content_type} - {content_size:,} bytes]"

            if start_index > 0:
                if start_index >= len(content):
                    content = "[Start index exceeds content length. No content available.]"
                else:
                    content = content[start_index:]
                    metadata += f"[Showing content from character {start_index:,} onwards]\n\n"

            if len(content) > max_length:
                content = content[:max_length]
                content += f"\n\n[Content truncated at {max_length:,} characters. Use start_index={max_length} to continue reading.]"

            if use_cache:
                _cache.set(cache_key, content, ttl=cache_ttl)
                logger.info(f"Cached result for URL: {url} (TTL: {cache_ttl}s)")

            formatted_content = metadata + content
            logger.info(f"Successfully fetched and processed URL: {url} ({content_size:,} bytes)")

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            webmcp_metrics["fetch_count"] += 1
            webmcp_metrics["total_fetch_time_ms"] += elapsed_ms
            if elapsed_ms < webmcp_metrics["min_fetch_time_ms"]:
                webmcp_metrics["min_fetch_time_ms"] = elapsed_ms
            if elapsed_ms > webmcp_metrics["max_fetch_time_ms"]:
                webmcp_metrics["max_fetch_time_ms"] = elapsed_ms
            if fef_manager is not None:
                fef_manager.metrics.record_request(
                    endpoint="tools/call", tool_name="fetch_url",
                    success=True, duration_ms=elapsed_ms
                )

            return formatted_content
    except httpx.TimeoutException:
        error_msg = f"Request timed out after {timeout} seconds"
        logger.error(f"Timeout error for URL {url}: {error_msg}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        webmcp_metrics["fetch_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="fetch_url",
                success=False, duration_ms=elapsed_ms
            )

        return f"Error: {error_msg}"
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code} error: {e.response.reason_phrase}"
        logger.error(f"HTTP error for URL {url}: {error_msg}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        webmcp_metrics["fetch_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="fetch_url",
                success=False, duration_ms=elapsed_ms
            )

        return f"Error: {error_msg}"
    except httpx.RequestError as e:
        error_msg = f"Network error: {str(e)}"
        logger.error(f"Network error for URL {url}: {error_msg}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        webmcp_metrics["fetch_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="fetch_url",
                success=False, duration_ms=elapsed_ms
            )

        return f"Error: {error_msg}"
    except ValueError as e:
        error_msg = f"Invalid URL or parameter: {str(e)}"
        logger.error(f"Value error for URL {url}: {error_msg}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        webmcp_metrics["fetch_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="fetch_url",
                success=False, duration_ms=elapsed_ms
            )

        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected error for URL {url}: {error_msg}", exc_info=True)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        webmcp_metrics["fetch_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="fetch_url",
                success=False, duration_ms=elapsed_ms
            )

        return f"Error: {error_msg}"
# ============================================================================
# FEF V3 Extensions Setup
# ============================================================================

fef_manager = None
fef_registry = None
fef_http_server = None
fef_setup_done = False


def setup_extensions(registry=None) -> None:
    """Set up FEF V3 extensions. Called by launcher or on startup."""
    global fef_manager, fef_registry, fef_http_server, fef_setup_done
    
    if fef_setup_done:
        return
    
    if not FEF_V3_AVAILABLE:
        fef_setup_done = True
        return
    
    mgmt_port = int(os.environ.get("MCP_MGMT_PORT", MGMT_PORT))
    
    custom_extensions = [
        Extension(
            name="search_stats",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {"type": "object", "properties": {"total_searches": {"type": "integer"}, "search_errors": {"type": "integer"}, "avg_search_time_ms": {"type": "number"}}}
            },
            handler=get_search_stats,
            metadata={"description": "Search engine statistics", "category": "metrics"}
        ),
        Extension(
            name="fetch_stats",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {"type": "object", "properties": {"total_fetches": {"type": "integer"}, "fetch_errors": {"type": "integer"}, "avg_fetch_time_ms": {"type": "number"}}}
            },
            handler=get_fetch_stats,
            metadata={"description": "URL fetch statistics", "category": "metrics"}
        ),
        Extension(
            name="search_history",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}}},
                "output": {"type": "object", "properties": {"recent_searches": {"type": "array", "items": {"type": "string"}}, "total": {"type": "integer"}}}
            },
            handler=get_search_history,
            metadata={"description": "Recent search queries", "category": "metrics"}
        ),
        Extension(
            name="fetch_cache_hits",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {"type": "object", "properties": {"cache_hits": {"type": "integer"}, "cache_misses": {"type": "integer"}, "hit_ratio": {"type": "number"}}}
            },
            handler=get_fetch_cache_hits,
            metadata={"description": "Fetch cache hit ratio", "category": "metrics"}
        ),
    ]
    
    if registry is not None:
        fef_registry = registry
        fef_manager = ToolExtensionManager(TOOL_NAME)
        register_common_extensions(TOOL_NAME, fef_registry, fef_manager)
        for ext in custom_extensions:
            fef_registry.register(TOOL_NAME, ext)
        fef_http_server = None
        logger.info(f"[{TOOL_NAME}] FEF V3 registered with launcher's registry")
    else:
        fef_manager, fef_registry, fef_http_server = setup_tool_extensions(
            tool_name=TOOL_NAME,
            mgmt_port=mgmt_port,
            custom_extensions=custom_extensions
        )
        logger.info(f"[{TOOL_NAME}] FEF V3 standalone mode on port {mgmt_port}")
    
    fef_setup_done = True


# ============================================================================
# Lifespan
# ============================================================================


@asynccontextmanager
async def lifespan(app):
    """Lifespan context manager for startup/shutdown."""
    logger.info(f"{TOOL_NAME} FastMCP server starting on port {MCP_PORT}...")
    
    if not fef_setup_done:
        setup_extensions(registry=None)
    
    if FEF_V3_AVAILABLE and fef_http_server:
        try:
            await fef_http_server.start()
            logger.info("FEF V3 management server started")
        except Exception as e:
            logger.warning(f"Failed to start FEF V3 management server: {e}")
    
    yield
    
    logger.info(f"{TOOL_NAME} FastMCP server shutting down...")
    if fef_http_server:
        try:
            await fef_http_server.stop()
        except Exception:
            pass
# ============================================================================
# App Export
# Transports: streamable-http (default, /mcp) or sse (legacy compat, /sse + /messages)
# ============================================================================

from tools.shared.server_factory import get_transport_app

app = get_transport_app(mcp)


# ============================================================================
# Exports for Launcher
# ============================================================================

__all__ = ["app", "setup_extensions", "mcp"]


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    transport = os.environ.get("MCP_TRANSPORT", "streamable-http").lower()
    logger.info(f"Starting {TOOL_NAME} FastMCP server (transport: {transport})")
    logger.info(f"  MCP port: {MCP_PORT}")
    if transport == "sse":
        logger.info(f"  SSE endpoint: http://localhost:{MCP_PORT}/sse")
    else:
        logger.info(f"  Streamable HTTP: http://localhost:{MCP_PORT}/mcp")
    if FEF_V3_AVAILABLE:
        logger.info(f"  FEF V3 mgmt: http://localhost:{MGMT_PORT}")
    
    uvicorn.run(app, host=DEFAULT_HOST, port=MCP_PORT, log_level="info", lifespan="on")
