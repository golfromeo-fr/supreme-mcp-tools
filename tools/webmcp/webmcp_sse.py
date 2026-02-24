#!/usr/bin/env python3
"""
Enhanced MCP Server with HTTP/SSE transport
Provides enhanced web search and URL fetch capabilities with content processing

Based on best practices from:
- mcp-server-fetch (official MCP fetch server)
- html2md-mcp (community HTML to Markdown converter)
"""
import sys
import os
import hashlib
from functools import lru_cache
from typing import Optional
from pathlib import Path

# Check for required dependencies before importing
try:
    import anyio
    import mcp.types as types
    from mcp.server.lowlevel import Server
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount
    import httpx
    import logging
    from urllib.parse import urlencode, urlparse
    import time
    import re
    from bs4 import BeautifulSoup
    from dotenv import load_dotenv
    from serpapi import GoogleSearch
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}", file=sys.stderr)
    print("Please make sure the virtual environment is activated and all dependencies are installed.", file=sys.stderr)
    print("Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("web_mcp")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("web_mcp")

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")

# Verify server components
logger.info("Initializing Web MCP Server...")

try:
    server = Server("web_mcp")
    sse_transport = SseServerTransport("/messages/")
    logger.info("Server components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize server components: {e}")
    sys.exit(1)


# ============================================================================
# Caching System (from html2md-mcp best practices)
# ============================================================================

class SimpleCache:
    """Simple in-memory cache with TTL support"""
    def __init__(self, default_ttl: int = 3600):
        self.cache: dict[str, tuple[str, float]] = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            content, expiry = self.cache[key]
            if time.time() < expiry:
                return content
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        expiry = time.time() + (ttl if ttl is not None else self.default_ttl)
        self.cache[key] = (value, expiry)
    
    def clear(self) -> None:
        self.cache.clear()
    
    def cleanup_expired(self) -> None:
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, expiry) in self.cache.items()
            if current_time >= expiry
        ]
        for key in expired_keys:
            del self.cache[key]


# Global cache instance
_cache = SimpleCache(default_ttl=3600)


def _generate_cache_key(url: str, params: dict) -> str:
    """Generate a cache key from URL and parameters"""
    param_str = str(sorted(params.items()))
    combined = f"{url}:{param_str}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


# ============================================================================
# HTML Processing Utilities (from html2md-mcp best practices)
# ============================================================================

def _clean_html(html_content: str,
               include_images: bool = True,
               include_tables: bool = True,
               include_links: bool = True) -> str:
    """
    Clean HTML by removing unnecessary elements
    
    Based on html2md-mcp implementation
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style tags
    for tag in soup(['script', 'style', 'noscript', 'iframe']):
        tag.decompose()
    
    # Remove navigation, header, footer elements (common patterns)
    for selector in [
        'nav', 'header', 'footer',
        '[role="navigation"]',
        '[role="banner"]',
        '[role="contentinfo"]',
        '.navigation', '.nav', '.menu', '.sidebar',
        '.footer', '.header', '.cookie-banner',
        '.advertisement', '.ads', '.promo'
    ]:
        for element in soup.select(selector):
            element.decompose()
    
    # Remove images if not requested
    if not include_images:
        for img in soup.find_all('img'):
            img.decompose()
    
    # Process tables if not requested
    if not include_tables:
        for table in soup.find_all('table'):
            table.decompose()
    
    # Process links if not requested
    if not include_links:
        for a in soup.find_all('a'):
            a.replace_with(a.get_text())
    
    return str(soup)


def _html_to_markdown(html_content: str) -> str:
    """
    Convert HTML to Markdown using BeautifulSoup (basic implementation)
    For production, consider using trafilatura or markdownify libraries
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Basic HTML to Markdown conversion
    markdown = []
    
    for element in soup.descendants:
        if element.name == 'h1':
            markdown.append(f"\n# {element.get_text().strip()}\n")
        elif element.name == 'h2':
            markdown.append(f"\n## {element.get_text().strip()}\n")
        elif element.name == 'h3':
            markdown.append(f"\n### {element.get_text().strip()}\n")
        elif element.name == 'p':
            text = element.get_text().strip()
            if text:
                markdown.append(f"\n{text}\n")
        elif element.name == 'strong' or element.name == 'b':
            markdown.append(f"**{element.get_text()}**")
        elif element.name == 'em' or element.name == 'i':
            markdown.append(f"*{element.get_text()}*")
        elif element.name == 'a':
            href = element.get('href', '')
            text = element.get_text()
            if href and text:
                markdown.append(f"[{text}]({href})")
        elif element.name == 'ul':
            markdown.append("\n")
        elif element.name == 'ol':
            markdown.append("\n")
        elif element.name == 'li':
            markdown.append(f"- {element.get_text().strip()}")
        elif element.name == 'br':
            markdown.append("\n")
        elif element.name == 'hr':
            markdown.append("\n---\n")
    
    return ''.join(markdown)


def _extract_text_content(html_content: str) -> str:
    """Extract plain text from HTML"""
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator='\n', strip=True)


# Define SSE handler
async def handle_sse(request):
    from starlette.responses import Response
    
    async with sse_transport.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())
    return Response()


# Tool: Enhanced Brave Web Search (Web Scraping)
async def brave_search_web_tool(arguments: dict) -> list[types.TextContent]:
    """Handler for Brave web search with enhanced features (web scraping)"""
    query = arguments.get("query")
    count = arguments.get("count", 10)
    timeout = float(arguments.get("timeout", 30.0))
    language = arguments.get("language", "en")

    if not query:
        return [types.TextContent(type="text", text="Error: 'query' parameter is required")]

    try:
        # Build Brave Search URL with enhanced parameters
        params = {
            'q': query,
            'count': str(count),
            'source': 'web',
            'lang': language
        }
        search_url = f"https://search.brave.com/search?{urlencode(params)}"

        # Enhanced browser-like headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": f"{language},en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
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

            # Process and format results
            content = response.text
            if len(content) > 100000:
                content = content[:100000] + "\n\n[Content truncated due to size limit]"

            # Add metadata header
            metadata = f"## Search Results Metadata\n- Query: {query}\n- Results Count: {count}\n- Language: {language}\n- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            formatted_content = metadata + content

            logger.info(f"Enhanced Brave search completed for query: {query}")
            return [types.TextContent(type="text", text=formatted_content)]

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code} error: {e.response.reason_phrase}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]
    except httpx.RequestError as e:
        error_msg = f"Network error: {str(e)}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]


# Tool: Brave Search API
async def brave_search_api_tool(arguments: dict) -> list[types.TextContent]:
    """
    Handler for Brave Search API with enhanced features
    
    Supports both GET and POST methods to the /res/v1/web/search endpoint.
    Returns formatted results including web, news, videos, discussions, infobox, FAQ, and locations.
    
    llm_mode: When True, returns concise JSON-like structure optimized for LLM consumption.
    """
    query = arguments.get("query")
    count = int(arguments.get("count", 10))
    timeout = float(arguments.get("timeout", 30.0))
    language = arguments.get("language", "en")
    country = arguments.get("country", "US")
    text_decorations = arguments.get("text_decorations", True)
    fresh = arguments.get("fresh", False)
    use_post = arguments.get("use_post", False)  # Use POST method instead of GET
    llm_mode = arguments.get("llm_mode", False)  # LLM-friendly mode

    if not query:
        return [types.TextContent(type="text", text="Error: 'query' parameter is required")]

    if not BRAVE_SEARCH_API_KEY:
        return [types.TextContent(type="text", text="Error: BRAVE_SEARCH_API_KEY not found in .env file. Please set your API key in the .env file.")]

    try:
        # Use the correct endpoint: /res/v1/web/search
        base_url = "https://api.search.brave.com/res/v1/web/search"
        
        # API headers
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": BRAVE_SEARCH_API_KEY
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            if use_post:
                # POST method with JSON body
                headers["Content-Type"] = "application/json"
                body = {
                    "q": query,
                    "count": min(max(count, 1), 20),  # Ensure count is between 1-20
                    "country": country,
                    "search_lang": language,
                    "text_decorations": text_decorations,
                    "fresh": fresh
                }
                response = await client.post(base_url, headers=headers, json=body)
            else:
                # GET method with query parameters
                params = {
                    "q": query,
                    "count": min(max(count, 1), 20),  # Ensure count is between 1-20
                    "country": country,
                    "search_lang": language,
                    "text_decorations": "true" if text_decorations else "false",
                    "fresh": "true" if fresh else "false"
                }
                response = await client.get(base_url, headers=headers, params=params)
            
            response.raise_for_status()

            # Process and format results
            result_data = response.json()

            # LLM mode: Return concise JSON-like structure
            if llm_mode:
                import json
                llm_results = {"query": query, "results": []}
                
                # Add web results
                if "web" in result_data and "results" in result_data["web"]:
                    for result in result_data["web"]["results"]:
                        llm_results["results"].append({
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "description": result.get("description", ""),
                            "type": "web"
                        })
                
                # Add news results
                if "news" in result_data and "results" in result_data["news"]:
                    for result in result_data["news"]["results"]:
                        llm_results["results"].append({
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "description": result.get("description", ""),
                            "age": result.get("age", ""),
                            "type": "news"
                        })
                
                # Add video results
                if "videos" in result_data and "results" in result_data["videos"]:
                    for result in result_data["videos"]["results"]:
                        video_info = result.get("video", {})
                        llm_results["results"].append({
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "description": result.get("description", ""),
                            "duration": video_info.get("duration", ""),
                            "views": video_info.get("views", 0),
                            "type": "video"
                        })
                
                # Add FAQ results
                if "faq" in result_data and "results" in result_data["faq"]:
                    faq_list = []
                    for result in result_data["faq"]["results"]:
                        faq_list.append({
                            "question": result.get("question", ""),
                            "answer": result.get("answer", ""),
                            "url": result.get("url", "")
                        })
                    llm_results["faq"] = faq_list
                
                logger.info(f"Brave Search API completed for query: {query} (llm_mode)")
                return [types.TextContent(type="text", text=json.dumps(llm_results, indent=2))]

            # Verbose mode: Human-readable format
            formatted_results = []
            formatted_results.append(f"## Brave Search API Results\n")
            formatted_results.append(f"- Query: {query}\n")
            formatted_results.append(f"- Results Count: {count}\n")
            formatted_results.append(f"- Language: {language}\n")
            formatted_results.append(f"- Country: {country}\n")
            formatted_results.append(f"- Method: {'POST' if use_post else 'GET'}\n")
            formatted_results.append(f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Process query metadata
            if "query" in result_data:
                query_info = result_data["query"]
                formatted_results.append("### Query Information\n\n")
                if "original" in query_info:
                    formatted_results.append(f"- Original Query: {query_info['original']}\n")
                if "cleaned" in query_info:
                    formatted_results.append(f"- Cleaned Query: {query_info['cleaned']}\n")
                if "altered" in query_info:
                    formatted_results.append(f"- Altered Query: {query_info['altered']}\n")
                if "language" in query_info and "main" in query_info["language"]:
                    formatted_results.append(f"- Detected Language: {query_info['language']['main']}\n")
                if "spellcheck_off" in query_info:
                    formatted_results.append(f"- Spellcheck: {'Off' if query_info['spellcheck_off'] else 'On'}\n")
                formatted_results.append("\n")

            # Process web results
            if "web" in result_data and "results" in result_data["web"]:
                formatted_results.append("### Web Results\n\n")
                for i, result in enumerate(result_data["web"]["results"], 1):
                    formatted_results.append(f"{i}. **{result.get('title', 'No title')}**\n")
                    formatted_results.append(f"   URL: {result.get('url', 'No URL')}\n")
                    formatted_results.append(f"   Description: {result.get('description', 'No description')}\n")
                    if "language" in result:
                        formatted_results.append(f"   Language: {result.get('language', 'Unknown')}\n")
                    if "profile" in result and "name" in result["profile"]:
                        formatted_results.append(f"   Source: {result['profile'].get('name', 'Unknown')}\n")
                    formatted_results.append("\n")

            # Process news results if available
            if "news" in result_data and "results" in result_data["news"]:
                formatted_results.append("### News Results\n\n")
                for i, result in enumerate(result_data["news"]["results"], 1):
                    formatted_results.append(f"{i}. **{result.get('title', 'No title')}**\n")
                    formatted_results.append(f"   URL: {result.get('url', 'No URL')}\n")
                    formatted_results.append(f"   Description: {result.get('description', 'No description')}\n")
                    if "age" in result:
                        formatted_results.append(f"   Age: {result.get('age', 'Unknown')}\n")
                    if "profile" in result and "name" in result["profile"]:
                        formatted_results.append(f"   Source: {result['profile'].get('name', 'Unknown')}\n")
                    formatted_results.append("\n")

            # Process video results if available
            if "videos" in result_data and "results" in result_data["videos"]:
                formatted_results.append("### Video Results\n\n")
                for i, result in enumerate(result_data["videos"]["results"], 1):
                    formatted_results.append(f"{i}. **{result.get('title', 'No title')}**\n")
                    formatted_results.append(f"   URL: {result.get('url', 'No URL')}\n")
                    formatted_results.append(f"   Description: {result.get('description', 'No description')}\n")
                    if "video" in result:
                        video_info = result["video"]
                        if "duration" in video_info:
                            formatted_results.append(f"   Duration: {video_info.get('duration', 'Unknown')}\n")
                        if "views" in video_info:
                            formatted_results.append(f"   Views: {video_info.get('views', 0):,}\n")
                        if "creator" in video_info:
                            formatted_results.append(f"   Creator: {video_info.get('creator', 'Unknown')}\n")
                    formatted_results.append("\n")

            # Process discussion results if available
            if "discussions" in result_data and "results" in result_data["discussions"]:
                formatted_results.append("### Discussion Results\n\n")
                for i, result in enumerate(result_data["discussions"]["results"], 1):
                    formatted_results.append(f"{i}. **{result.get('title', 'No title')}**\n")
                    formatted_results.append(f"   URL: {result.get('url', 'No URL')}\n")
                    formatted_results.append(f"   Description: {result.get('description', 'No description')}\n")
                    if "profile" in result and "name" in result["profile"]:
                        formatted_results.append(f"   Source: {result['profile'].get('name', 'Unknown')}\n")
                    formatted_results.append("\n")

            # Process infobox results if available
            if "infobox" in result_data and "results" in result_data["infobox"]:
                formatted_results.append("### Infobox Results\n\n")
                for i, result in enumerate(result_data["infobox"]["results"], 1):
                    formatted_results.append(f"{i}. **{result.get('title', 'No title')}**\n")
                    formatted_results.append(f"   URL: {result.get('url', 'No URL')}\n")
                    formatted_results.append(f"   Description: {result.get('description', 'No description')}\n")
                    formatted_results.append("\n")

            # Process FAQ results if available
            if "faq" in result_data and "results" in result_data["faq"]:
                formatted_results.append("### FAQ Results\n\n")
                for i, result in enumerate(result_data["faq"]["results"], 1):
                    formatted_results.append(f"{i}. **Question: {result.get('question', 'No question')}**\n")
                    formatted_results.append(f"   Answer: {result.get('answer', 'No answer')}\n")
                    if "url" in result:
                        formatted_results.append(f"   Source: {result.get('url', 'No URL')}\n")
                    formatted_results.append("\n")

            # Process location results if available
            if "locations" in result_data and "results" in result_data["locations"]:
                formatted_results.append("### Location Results\n\n")
                for i, result in enumerate(result_data["locations"]["results"], 1):
                    formatted_results.append(f"{i}. **{result.get('title', 'No title')}**\n")
                    formatted_results.append(f"   URL: {result.get('url', 'No URL')}\n")
                    formatted_results.append(f"   Description: {result.get('description', 'No description')}\n")
                    formatted_results.append("\n")

            # Process summarizer if available
            if "summarizer" in result_data:
                formatted_results.append("### AI Summary\n\n")
                summarizer = result_data["summarizer"]
                if "key" in summarizer:
                    formatted_results.append(f"Summary Key: {summarizer['key']}\n")
                formatted_results.append("\n")

            # Process rich results if available
            if "rich" in result_data and "hint" in result_data["rich"]:
                formatted_results.append("### Rich Results\n\n")
                hint = result_data["rich"]["hint"]
                if "vertical" in hint:
                    formatted_results.append(f"Type: {hint['vertical']}\n")
                if "callback_key" in hint:
                    formatted_results.append(f"Callback Key: {hint['callback_key']}\n")
                formatted_results.append("\n")

            formatted_content = ''.join(formatted_results)

            logger.info(f"Brave Search API completed for query: {query} (method: {'POST' if use_post else 'GET'})")
            return [types.TextContent(type="text", text=formatted_content)]

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code} error: {e.response.reason_phrase}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]
    except httpx.RequestError as e:
        error_msg = f"Network error: {str(e)}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]


# Tool: Google Search API (SerpAPI)
async def google_search_api_tool(arguments: dict) -> list[types.TextContent]:
    """
    Handler for Google Search API using SerpAPI
    
    Supports Google Search with various parameters including location, language, pagination, and safe search.
    Returns formatted results including organic results, answer box, knowledge graph, and more.
    
    llm_mode: When True, returns concise JSON-like structure optimized for LLM consumption.
    """
    query = arguments.get("query")
    engine = arguments.get("engine", "google")
    google_domain = arguments.get("google_domain", "google.com")
    hl = arguments.get("hl", "en")  # Language
    gl = arguments.get("gl", "us")  # Country
    location = arguments.get("location", None)  # Location for localized results
    start = int(arguments.get("start", 0))  # Pagination offset
    num = int(arguments.get("num", 10))  # Number of results (1-100)
    safe = arguments.get("safe", "active")  # Safe search: active, off
    device = arguments.get("device", None)  # Device: desktop, mobile, tablet
    llm_mode = arguments.get("llm_mode", False)  # LLM-friendly mode

    if not query:
        return [types.TextContent(type="text", text="Error: 'query' parameter is required")]

    if not SERPAPI_API_KEY:
        return [types.TextContent(type="text", text="Error: SERPAPI_API_KEY not found in .env file. Please set your API key in the .env file.")]

    try:
        # Build SerpAPI parameters
        params = {
            "engine": engine,
            "q": query,
            "google_domain": google_domain,
            "hl": hl,
            "gl": gl,
            "api_key": SERPAPI_API_KEY,
            "num": min(max(num, 1), 100)  # Ensure num is between 1-100
        }

        # Add optional parameters
        if location:
            params["location"] = location
        if start > 0:
            params["start"] = str(start)
        if safe:
            params["safe"] = safe
        if device and device.strip():
            params["device"] = device

        # Execute search using SerpAPI
        search = GoogleSearch(params)
        results = search.get_dict()

        # Log the raw response for debugging
        logger.info(f"Raw SerpAPI response keys: {list(results.keys())}")

        # LLM mode: Return concise JSON-like structure
        if llm_mode:
            import json
            llm_results = {"query": query, "results": []}
            
            # Add organic results
            if "organic_results" in results:
                for result in results["organic_results"]:
                    llm_results["results"].append({
                        "title": result.get("title", ""),
                        "url": result.get("link", ""),
                        "snippet": result.get("snippet", "")
                    })
            
            # Add answer box if available
            if "answer_box" in results:
                ab = results["answer_box"]
                llm_results["answer"] = ab.get("answer", ab.get("snippet", ""))
            
            # Add knowledge graph if available
            if "knowledge_graph" in results:
                kg = results["knowledge_graph"]
                llm_results["knowledge_graph"] = {
                    "title": kg.get("title", ""),
                    "type": kg.get("type", ""),
                    "description": kg.get("description", ""),
                    "website": kg.get("website", "")
                }
            
            logger.info(f"Google Search API (SerpAPI) completed for query: {query} (llm_mode)")
            return [types.TextContent(type="text", text=json.dumps(llm_results, indent=2))]

        # Verbose mode: Human-readable format
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

        # Check for error response
        if "error" in results:
            error_info = results["error"]
            formatted_results.append("### Error\n\n")
            # Handle both dict and string error responses
            if isinstance(error_info, dict):
                formatted_results.append(f"Error Code: {error_info.get('code', 'Unknown')}\n")
                formatted_results.append(f"Error Message: {error_info.get('message', 'No message')}\n")
            else:
                formatted_results.append(f"Error: {str(error_info)}\n")
            formatted_content = ''.join(formatted_results)
            logger.error(f"SerpAPI error: {error_info}")
            return [types.TextContent(type="text", text=formatted_content)]

        # Process search information
        if "search_information" in results:
            search_info = results["search_information"]
            formatted_results.append("### Search Information\n\n")
            if "query_displayed" in search_info:
                formatted_results.append(f"- Query Displayed: {search_info['query_displayed']}\n")
            if "total_results" in search_info:
                formatted_results.append(f"- Total Results: {search_info['total_results']}\n")
            if "time_taken_displayed" in search_info:
                formatted_results.append(f"- Time Taken: {search_info['time_taken_displayed']}\n")
            formatted_results.append("\n")

        # Process answer box
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

        # Process knowledge graph
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
            if "images" in kg and len(kg["images"]) > 0:
                formatted_results.append(f"Images: {len(kg['images'])} available\n")
            formatted_results.append("\n")

        # Process organic results
        if "organic_results" in results:
            formatted_results.append("### Organic Results\n\n")
            for i, result in enumerate(results["organic_results"], 1):
                formatted_results.append(f"{i}. **{result.get('title', 'No title')}**\n")
                formatted_results.append(f"   URL: {result.get('link', 'No URL')}\n")
                if "snippet" in result:
                    formatted_results.append(f"   Description: {result['snippet']}\n")
                if "displayed_link" in result:
                    formatted_results.append(f"   Displayed URL: {result['displayed_link']}\n")
                if "date" in result:
                    formatted_results.append(f"   Date: {result['date']}\n")
                if "snippet_highlighted_words" in result:
                    formatted_results.append(f"   Highlighted: {', '.join(result['snippet_highlighted_words'])}\n")
                formatted_results.append("\n")

        # Process people also ask
        if "people_also_ask" in results:
            formatted_results.append("### People Also Ask\n\n")
            for i, paa in enumerate(results["people_also_ask"], 1):
                formatted_results.append(f"{i}. {paa.get('question', 'No question')}\n")
                if "snippet" in paa:
                    formatted_results.append(f"   {paa['snippet']}\n")
                if "link" in paa:
                    formatted_results.append(f"   Source: {paa['link']}\n")
                formatted_results.append("\n")

        # Process related searches
        if "related_searches" in results:
            formatted_results.append("### Related Searches\n\n")
            for i, related in enumerate(results["related_searches"], 1):
                formatted_results.append(f"{i}. {related.get('query', 'No query')}\n")
                if "link" in related:
                    formatted_results.append(f"   URL: {related['link']}\n")
                formatted_results.append("\n")

        # Process top stories (news)
        if "top_stories" in results:
            formatted_results.append("### Top Stories\n\n")
            for i, story in enumerate(results["top_stories"], 1):
                formatted_results.append(f"{i}. **{story.get('title', 'No title')}**\n")
                if "link" in story:
                    formatted_results.append(f"   URL: {story['link']}\n")
                if "source" in story:
                    formatted_results.append(f"   Source: {story['source']}\n")
                if "date" in story:
                    formatted_results.append(f"   Date: {story['date']}\n")
                if "thumbnail" in story:
                    formatted_results.append(f"   Thumbnail: {story['thumbnail']}\n")
                formatted_results.append("\n")

        # Process local results
        if "local_results" in results:
            formatted_results.append("### Local Results\n\n")
            for i, local in enumerate(results["local_results"], 1):
                formatted_results.append(f"{i}. **{local.get('title', 'No title')}**\n")
                if "address" in local:
                    formatted_results.append(f"   Address: {local['address']}\n")
                if "phone" in local:
                    formatted_results.append(f"   Phone: {local['phone']}\n")
                if "rating" in local:
                    formatted_results.append(f"   Rating: {local['rating']}\n")
                if "reviews" in local:
                    formatted_results.append(f"   Reviews: {local['reviews']}\n")
                if "links" in local and "website" in local["links"]:
                    formatted_results.append(f"   Website: {local['links']['website']}\n")
                formatted_results.append("\n")

        formatted_content = ''.join(formatted_results)

        logger.info(f"Google Search API (SerpAPI) completed for query: {query}")
        return [types.TextContent(type="text", text=formatted_content)]

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]


# ============================================================================
# Enhanced Web Reader Tool (with best practices from mcp-server-fetch and html2md-mcp)
# ============================================================================

async def post_url_tool(arguments: dict) -> list[types.TextContent]:
    """
    Handler for sending POST requests to a URL with JSON payload support.
    
    Based on oracleMCP implementation.
    Supports HTTP POST with optional data and headers.
    """
    logger.debug(f"Processing post_url tool with arguments: {arguments}")
    url = arguments.get("url")
    data = arguments.get("data")
    headers = arguments.get("headers", {})
    timeout = float(arguments.get("timeout", 30.0))
    
    if not url:
        logger.error("Missing required parameter: url")
        return [types.TextContent(type="text", text="Missing required parameter: url")]
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, data=data, headers=headers)
            response.raise_for_status()
            logger.info(f"Successfully posted to URL: {url}")
            
            # Build metadata
            metadata_lines = [
                "## POST URL Results",
                f"- URL: {url}",
                f"- Status Code: {response.status_code}",
                f"- Content Type: {response.headers.get('content-type', 'unknown')}",
                f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            ]
            if data:
                metadata_lines.append(f"- Data: {data[:200] if len(data) > 200 else data}...")
            metadata = '\n'.join(metadata_lines) + '\n\n'
            
            return [types.TextContent(type="text", text=metadata + response.text)]
    except httpx.HTTPError as e:
        error_msg = f"HTTP error occurred while posting to URL {url}: {str(e)}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=error_msg)]
    except Exception as e:
        error_msg = f"Error posting to URL {url}: {str(e)}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=error_msg)]


async def fetch_url_tool(arguments: dict) -> list[types.TextContent]:
    """
    Enhanced URL fetch tool with comprehensive content processing
    
    Based on best practices from:
    - mcp-server-fetch (official MCP fetch server)
    - html2md-mcp (community HTML to Markdown converter)
    
    Features:
    - Content size management (max_length, start_index for pagination)
    - Content filtering (images, tables, links)
    - Caching support (use_cache, cache_ttl)
    - Multiple output formats (markdown, text, raw)
    - Comprehensive error handling
    """
    # Required parameters
    url = arguments.get("url")
    if not url:
        return [types.TextContent(type="text", text="Error: 'url' parameter is required")]
    
    # Optional parameters with defaults
    timeout = float(arguments.get("timeout", 30.0))
    max_length = int(arguments.get("max_length", 50000))  # Character limit for output
    start_index = int(arguments.get("start_index", 0))  # For pagination/chunked reading
    output_format = arguments.get("format", "markdown")  # markdown, text, raw
    follow_redirects = arguments.get("follow_redirects", True)
    headers = arguments.get("headers", {})
    response_type = arguments.get("response_type", "auto").lower()  # auto, json, html, text
    
    # Content filtering options (from html2md-mcp)
    include_images = arguments.get("include_images", True)
    include_tables = arguments.get("include_tables", True)
    include_links = arguments.get("include_links", True)
    
    # Performance options (from html2md-mcp)
    use_cache = arguments.get("use_cache", False)
    cache_ttl = int(arguments.get("cache_ttl", 3600))  # Default 1 hour
    max_size = int(arguments.get("max_size", 10 * 1024 * 1024))  # Default 10MB
    
    # Legacy parameter support (for backward compatibility)
    retain_images = arguments.get("retainImages", include_images)
    no_cache = arguments.get("noCache", not use_cache)
    
    # Override with legacy parameters if provided
    if "retainImages" in arguments:
        include_images = retain_images
    if "noCache" in arguments:
        use_cache = not no_cache
    
    # Validate parameters
    if timeout < 1 or timeout > 300:
        return [types.TextContent(type="text", text="Error: 'timeout' must be between 1 and 300 seconds")]
    if max_length < 100 or max_length > 1000000:
        return [types.TextContent(type="text", text="Error: 'max_length' must be between 100 and 1000000 characters")]
    if max_size < 1024 or max_size > 50 * 1024 * 1024:
        return [types.TextContent(type="text", text="Error: 'max_size' must be between 1KB and 50MB")]
    
    # Check cache if enabled
    if use_cache:
        cache_key = _generate_cache_key(url, {
            'format': output_format,
            'include_images': include_images,
            'include_tables': include_tables,
            'include_links': include_links,
            'start_index': start_index,
            'max_length': max_length
        })
        cached_content = _cache.get(cache_key)
        if cached_content is not None:
            logger.info(f"Cache hit for URL: {url}")
            return [types.TextContent(
                type="text",
                text=f"## Web Reader Results (Cached)\n- URL: {url}\n- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n- Format: {output_format}\n\n{cached_content}"
            )]
    
    # Enhanced browser-like headers (from mcp-server-fetch)
    default_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }
    
    # Add cache control headers if not using cache
    if not use_cache:
        default_headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        default_headers["Pragma"] = "no-cache"
    
    # Merge custom headers with defaults
    merged_headers = {**default_headers, **headers}
    
    try:
        # Clean up expired cache entries periodically
        _cache.cleanup_expired()
        
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=follow_redirects,
            headers=merged_headers,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)
        ) as client:
            # Fetch with size limit
            response = await client.get(url)
            response.raise_for_status()
            
            # Check content size before processing
            content_size = len(response.content)
            if content_size > max_size:
                logger.warning(f"Content size {content_size} exceeds limit {max_size} for URL: {url}")
                return [types.TextContent(
                    type="text",
                    text=f"Error: Content size ({content_size:,} bytes) exceeds maximum allowed size ({max_size:,} bytes)."
                )]
            
            # Get content
            content = response.text
            content_type = response.headers.get("content-type", "").lower()
            
            # Build metadata
            metadata_lines = [
                "## Web Reader Results",
                f"- URL: {url}",
                f"- Content Type: {content_type}",
                f"- Status Code: {response.status_code}",
                f"- Content Size: {content_size:,} bytes",
                f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"- Format: {output_format}",
                f"- Include Images: {include_images}",
                f"- Include Tables: {include_tables}",
                f"- Include Links: {include_links}",
                f"- Start Index: {start_index}",
                f"- Max Length: {max_length:,} characters"
            ]
            metadata = '\n'.join(metadata_lines) + '\n\n'
            
            # Process content based on type (with auto-detection support)
            import json
            
            # Handle JSON response type
            if response_type == "json" or (response_type == "auto" and "application/json" in content_type):
                try:
                    json_data = response.json()
                    content = json.dumps(json_data, indent=2)
                    if output_format == "markdown":
                        content = f"```json\n{content}\n```"
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON response from {url}")
                    content = response.text
                    if output_format == "markdown":
                        content = f"```\n{content}\n```"
            
            # Handle HTML response type
            elif response_type == "html" or (response_type == "auto" and "text/html" in content_type):
                # Clean HTML based on filtering options
                content = _clean_html(
                    content,
                    include_images=include_images,
                    include_tables=include_tables,
                    include_links=include_links
                )
                
                # Convert based on output format
                if output_format == "markdown":
                    content = _html_to_markdown(content)
                    content = f"# Content from {url}\n\n{content}"
                elif output_format == "text":
                    content = _extract_text_content(content)
                # For "raw", keep the cleaned HTML as-is
            
            # Handle text response type
            elif response_type == "text" or (response_type == "auto" and "text/plain" in content_type):
                # Plain text - just return as-is
                if output_format == "markdown":
                    content = f"```\n{content}\n```"
            
            # For other content types or when response_type doesn't match
            else:
                # Binary or other content types
                content = f"[Binary content - {content_type} - {content_size:,} bytes]"
            
            # Apply start_index for pagination (from mcp-server-fetch)
            if start_index > 0:
                if start_index >= len(content):
                    content = "[Start index exceeds content length. No content available.]"
                else:
                    content = content[start_index:]
                    metadata += f"[Showing content from character {start_index:,} onwards]\n\n"
            
            # Apply max_length truncation (from mcp-server-fetch)
            if len(content) > max_length:
                content = content[:max_length]
                content += f"\n\n[Content truncated at {max_length:,} characters. Use start_index={max_length} to continue reading.]"
            
            # Cache the result if caching is enabled
            if use_cache:
                _cache.set(cache_key, content, ttl=cache_ttl)
                logger.info(f"Cached result for URL: {url} (TTL: {cache_ttl}s)")
            
            formatted_content = metadata + content
            logger.info(f"Successfully fetched and processed URL: {url} ({content_size:,} bytes)")
            return [types.TextContent(type="text", text=formatted_content)]
    
    except httpx.TimeoutException:
        error_msg = f"Request timed out after {timeout} seconds"
        logger.error(f"Timeout error for URL {url}: {error_msg}")
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code} error: {e.response.reason_phrase}"
        logger.error(f"HTTP error for URL {url}: {error_msg}")
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]
    except httpx.RequestError as e:
        error_msg = f"Network error: {str(e)}"
        logger.error(f"Network error for URL {url}: {error_msg}")
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]
    except ValueError as e:
        error_msg = f"Invalid URL or parameter: {str(e)}"
        logger.error(f"Value error for URL {url}: {error_msg}")
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected error for URL {url}: {error_msg}", exc_info=True)
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]


# List tools
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="brave_search_web",
            description="Enhanced web search using Brave Search with language support and metadata (web scraping). Returns formatted HTML results.",
            inputSchema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "count": {
                        "type": "number",
                        "description": "Number of results to return (default: 10)",
                        "default": 10
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Search timeout in seconds (default: 30.0)",
                        "default": 30.0
                    },
                    "language": {
                        "type": "string",
                        "description": "Search language code (default: 'en')",
                        "default": "en"
                    }
                }
            }
        ),
        types.Tool(
            name="brave_search_api",
            description="Enhanced web search using Brave Search API with structured results. Requires BRAVE_SEARCH_API_KEY in .env file. Returns formatted JSON results including web results, news, videos, discussions, infobox, FAQ, and locations. Supports both GET and POST methods. Use llm_mode=True for concise JSON output optimized for LLM consumption.",
            inputSchema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "count": {
                        "type": "number",
                        "description": "Number of results to return (1-20, default: 10)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Search timeout in seconds (default: 30.0)",
                        "default": 30.0
                    },
                    "language": {
                        "type": "string",
                        "description": "Search language code (default: 'en')",
                        "default": "en"
                    },
                    "country": {
                        "type": "string",
                        "description": "Country code for search (default: 'US')",
                        "default": "US"
                    },
                    "text_decorations": {
                        "type": "boolean",
                        "description": "Whether to include text decorations (default: True)",
                        "default": True
                    },
                    "fresh": {
                        "type": "boolean",
                        "description": "Whether to prefer fresh results (default: False)",
                        "default": False
                    },
                    "use_post": {
                        "type": "boolean",
                        "description": "Use POST method instead of GET (default: False). POST sends parameters in JSON body, GET sends as query parameters.",
                        "default": False
                    },
                    "llm_mode": {
                        "type": "boolean",
                        "description": "Return concise JSON-like structure optimized for LLM consumption (default: False). When True, returns minimal structured data without verbose formatting.",
                        "default": False
                    }
                }
            }
        ),
        types.Tool(
            name="google_search_api",
            description="Google Search API using SerpAPI. Requires SERPAPI_API_KEY in .env file. Returns formatted results including organic results, answer box, knowledge graph, people also ask, related searches, top stories, and local results. Use llm_mode=True for concise JSON output optimized for LLM consumption.",
            inputSchema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "engine": {
                        "type": "string",
                        "description": "Search engine to use (default: 'google')",
                        "default": "google"
                    },
                    "google_domain": {
                        "type": "string",
                        "description": "Google domain to use (default: 'google.com')",
                        "default": "google.com"
                    },
                    "hl": {
                        "type": "string",
                        "description": "Language code for search results (default: 'en')",
                        "default": "en"
                    },
                    "gl": {
                        "type": "string",
                        "description": "Country code for search results (default: 'us')",
                        "default": "us"
                    },
                    "location": {
                        "type": "string",
                        "description": "Location for localized search results (optional)"
                    },
                    "start": {
                        "type": "number",
                        "description": "Pagination offset (default: 0)",
                        "default": 0,
                        "minimum": 0
                    },
                    "num": {
                        "type": "number",
                        "description": "Number of results to return (1-100, default: 10)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 100
                    },
                    "safe": {
                        "type": "string",
                        "description": "Safe search setting: 'active' or 'off' (default: 'active')",
                        "default": "active"
                    },
                    "device": {
                        "type": "string",
                        "description": "Device type: 'desktop', 'mobile', or 'tablet' (optional)"
                    },
                    "llm_mode": {
                        "type": "boolean",
                        "description": "Return concise JSON-like structure optimized for LLM consumption (default: False). When True, returns minimal structured data without verbose formatting.",
                        "default": False
                    }
                }
            }
        ),
        types.Tool(
            name="fetch_url",
            description="Enhanced web reader that fetches and processes web content with pagination, caching, and content filtering. Based on best practices from mcp-server-fetch and html2md-mcp. Supports auto-detection of response type (json/html/text) via response_type parameter.",
            inputSchema={
                "type": "object",
                "required": ["url"],
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch and process"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Request timeout in seconds (default: 30.0, range: 1-300)",
                        "default": 30.0,
                        "minimum": 1,
                        "maximum": 300
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional custom HTTP headers"
                    },
                    "follow_redirects": {
                        "type": "boolean",
                        "description": "Whether to follow redirects (default: True)",
                        "default": True
                    },
                    "format": {
                        "type": "string",
                        "description": "Output format (default: 'markdown')",
                        "enum": ["markdown", "text", "raw"],
                        "default": "markdown"
                    },
                    "response_type": {
                        "type": "string",
                        "description": "Response type handling (default: 'auto'). 'auto' detects from Content-Type header. Options: 'auto', 'json', 'html', 'text'.",
                        "enum": ["auto", "json", "html", "text"],
                        "default": "auto"
                    },
                    "max_length": {
                        "type": "number",
                        "description": "Maximum number of characters to return (default: 50000, range: 100-1000000). For pagination, use start_index parameter.",
                        "default": 50000,
                        "minimum": 100,
                        "maximum": 1000000
                    },
                    "start_index": {
                        "type": "number",
                        "description": "Start content from this character index for pagination (default: 0). Use with max_length to read large pages in chunks.",
                        "default": 0,
                        "minimum": 0
                    },
                    "include_images": {
                        "type": "boolean",
                        "description": "Whether to include images in output (default: True)",
                        "default": True
                    },
                    "include_tables": {
                        "type": "boolean",
                        "description": "Whether to include tables in output (default: True)",
                        "default": True
                    },
                    "include_links": {
                        "type": "boolean",
                        "description": "Whether to include links in output (default: True)",
                        "default": True
                    },
                    "max_size": {
                        "type": "number",
                        "description": "Maximum download size in bytes (default: 10485760 = 10MB, range: 1024-52428800 = 1KB-50MB)",
                        "default": 10485760,
                        "minimum": 1024,
                        "maximum": 52428800
                    },
                    "use_cache": {
                        "type": "boolean",
                        "description": "Enable caching for repeated URL requests (default: False)",
                        "default": False
                    },
                    "cache_ttl": {
                        "type": "number",
                        "description": "Cache time-to-live in seconds (default: 3600 = 1 hour, range: 60-86400)",
                        "default": 3600,
                        "minimum": 60,
                        "maximum": 86400
                    },
                    "retainImages": {
                        "type": "boolean",
                        "description": "Legacy parameter: Whether to retain images in output (default: True). Use include_images instead.",
                        "default": True
                    },
                    "noCache": {
                        "type": "boolean",
                        "description": "Legacy parameter: Disable caching (default: False). Use use_cache instead.",
                        "default": False
                    }
                }
            }
        ),
        types.Tool(
            name="post_url",
            description="Sends an HTTP POST request to the specified URL with optional data and headers. Supports JSON payloads and returns the response as text.",
            inputSchema={
                "type": "object",
                "required": ["url"],
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to POST to."
                    },
                    "data": {
                        "type": "string",
                        "description": "The POST body (as a string, e.g. JSON or form data)."
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers as a JSON object."
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Request timeout in seconds (default: 30.0)",
                        "default": 30.0
                    }
                }
            }
        )
    ]


# Tool router
@server.call_tool()
async def tool_router(name: str, arguments: dict) -> list[types.TextContent]:
    """Route tool calls to appropriate handlers"""
    logger.info(f"Tool called: {name} with arguments: {arguments}")

    handlers = {
        "brave_search_web": brave_search_web_tool,
        "brave_search_api": brave_search_api_tool,
        "google_search_api": google_search_api_tool,
        "fetch_url": fetch_url_tool,
        "post_url": post_url_tool
    }

    handler = handlers.get(name)
    if handler:
        return await handler(arguments)

    logger.error(f"Unknown tool: {name}")
    raise ValueError(f"Unknown tool: {name}")


# Resource handlers (empty for now)
@server.list_resources()
async def list_resources() -> list[types.Resource]:
    return []


@server.list_resource_templates()
async def list_resource_templates() -> list[types.ResourceTemplate]:
    return []


# Create Starlette app
app = Starlette(
    debug=True,
    routes=[
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Mount("/messages/", app=sse_transport.handle_post_message),
    ]
)


# Start the server
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Enhanced Web MCP Server on http://0.0.0.0:8001")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except KeyboardInterrupt:
        logger.info("Server shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


"""
VSCode settings.json configuration:

{
  "mcpServers": {
    "web_mcp": {
        "type": "sse",
        "url": "http://localhost:8001/sse",
        "headers": {
          "Content-Type": "application/json"
        }
      }
    }
  }
}
"""
