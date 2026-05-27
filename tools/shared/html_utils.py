"""
Shared HTML processing utilities for FastMCP tools.
"""

import re


# Compiled regex patterns for performance
_SCRIPT_STYLE_PATTERN = re.compile(r'<(script|style|noscript|iframe)[^>]*>.*?</\1>', re.DOTALL | re.IGNORECASE)
_NAV_FOOTER_PATTERN = re.compile(r'<(nav|header|footer|aside)[^>]*>.*?</\1>', re.DOTALL | re.IGNORECASE)
_COMMENT_PATTERN = re.compile(r'<!--.*?-->', re.DOTALL)
_WHITESPACE_PATTERN = re.compile(r'\s+')


def clean_html_basic(html_content: str, include_images: bool = True, 
                    include_tables: bool = True, include_links: bool = True) -> str:
    """
    Basic HTML cleaning without BeautifulSoup dependency for better performance.
    
    Args:
        html_content: Raw HTML content
        include_images: Whether to keep image tags
        include_tables: Whether to keep table tags
        include_links: Whether to keep link tags
        
    Returns:
        Cleaned text content
    """
    if not html_content:
        return ""
    
    # Remove script, style, noscript, iframe tags
    text = _SCRIPT_STYLE_PATTERN.sub('', html_content)
    
    # Remove navigation, header, footer, aside tags (common boilerplate)
    if not include_tables:  # Only remove nav/footer if not keeping tables for structure
        text = _NAV_FOOTER_PATTERN.sub('', text)
    
    # Remove HTML comments
    text = _COMMENT_PATTERN.sub('', text)
    
    # Handle images
    if not include_images:
        text = re.sub(r'<img[^>]*>', '', text, flags=re.IGNORECASE)
    
    # Handle tables
    if not include_tables:
        text = re.sub(r'<table[^>]*>.*?</table>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<(thead|tbody|tfoot|tr|td|th)[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</(thead|tbody|tfoot|tr|td|th)>', '', text, flags=re.IGNORECASE)
    
    # Handle links
    if not include_links:
        # Replace <a> tags with their text content
        text = re.sub(r'<a[^>]*>(.*?)</a>', r'\1', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove any remaining link tags
        text = re.sub(r'</?a[^>]*>', '', text, flags=re.IGNORECASE)
    
    # Remove all other HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up whitespace
    text = _WHITESPACE_PATTERN.sub(' ', text)
    return text.strip()


def html_to_markdown_simple(html_content: str) -> str:
    """
    Simple HTML to Markdown conversion without BeautifulSoup.
    
    Args:
        html_content: Raw HTML content
        
    Returns:
        Markdown formatted text
    """
    if not html_content:
        return ""
    
    text = html_content
    
    # Handle headers
    text = re.sub(r'<h1[^>]*>(.*?)</h1>', r'\n# \1\n', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<h2[^>]*>(.*?)</h2>', r'\n## \1\n', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<h3[^>]*>(.*?)</h3>', r'\n### \1\n', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<h4[^>]*>(.*?)</h4>', r'\n#### \1\n', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<h5[^>]*>(.*?)</h5>', r'\n##### \1\n', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<h6[^>]*>(.*?)</h6>', r'\n###### \1\n', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Handle paragraphs
    text = re.sub(r'<p[^>]*>(.*?)</p>', r'\n\1\n', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Handle line breaks
    text = re.sub(r'<br[^>]*/?>', r'\n', text, flags=re.IGNORECASE)
    
    # Handle strong/bold
    text = re.sub(r'<(strong|b)[^>]*>(.*?)</\1>', r'**\2**', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Handle emphasis/italic
    text = re.sub(r'<(em|i)[^>]*>(.*?)</\1>', r'*\2*', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Handle lists
    text = re.sub(r'<ul[^>]*>', r'\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<ol[^>]*>', r'\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<li[^>]*>(.*?)</li>', r'  - \1\n', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'</?(ul|ol)>', r'\n', text, flags=re.IGNORECASE)
    
    # Remove all remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\n\s*\n', r'\n\n', text)  # Multiple blank lines to double
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
    return text.strip()


def extract_text_content(html_content: str) -> str:
    """
    Extract plain text content from HTML.
    
    Args:
        html_content: Raw HTML content
        
    Returns:
        Plain text content
    """
    if not html_content:
        return ""
    
    # Remove script and style elements
    text = re.sub(r'<(script|style|noscript|iframe)[^>]*>.*?</\1>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove all other HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Global compiled regex patterns for performance
_COMPILED_PATTERNS = {
    'script_style': re.compile(r'<(script|style|noscript|iframe)[^>]*>.*?</\1>', re.DOTALL | re.IGNORECASE),
    'nav_footer': re.compile(r'<(nav|header|footer|aside)[^>]*>.*?</\1>', re.DOTALL | re.IGNORECASE),
    'comments': re.compile(r'<!--.*?-->', re.DOTALL),
    'whitespace': re.compile(r'\s+'),
    'html_tags': re.compile(r'<[^>]+>'),
    'img_tags': re.compile(r'<img[^>]*>', re.IGNORECASE),
    'table_tags': re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE),
    'link_tags': re.compile(r'<a[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE),
}


def clean_html_optimized(html_content: str, include_images: bool = True, 
                        include_tables: bool = True, include_links: bool = True) -> str:
    """
    Optimized HTML cleaning using pre-compiled regex patterns.
    
    Args:
        html_content: Raw HTML content
        include_images: Whether to keep image tags
        include_tables: Whether to keep table tags
        include_links: Whether to keep link tags
        
    Returns:
        Cleaned text content
    """
    if not html_content:
        return ""
    
    patterns = _COMPILED_PATTERNS
    
    # Remove script, style, noscript, iframe tags
    text = patterns['script_style'].sub('', html_content)
    
    # Remove HTML comments
    text = patterns['comments'].sub('', text)
    
    # Remove navigation, header, footer, aside tags (common boilerplate)
    if not include_tables:
        text = patterns['nav_footer'].sub('', text)
    
    # Handle images
    if not include_images:
        text = patterns['img_tags'].sub('', text)
    
    # Handle tables
    if not include_tables:
        text = patterns['table_tags'].sub('', text)
        # Remove table structural elements
        text = re.sub(r'</?(thead|tbody|tfoot|tr|td|th)[^>]*>', '', text, flags=re.IGNORECASE)
    
    # Handle links
    if not include_links:
        # Replace <a> tags with their text content
        text = patterns['link_tags'].sub(r'\1', text)
        # Remove any remaining anchor tags
        text = re.sub(r'</?a[^>]*>', '', text, flags=re.IGNORECASE)
    
    # Remove all other HTML tags
    text = patterns['html_tags'].sub('', text)
    
    # Clean up whitespace
    text = patterns['whitespace'].sub(' ', text)
    return text.strip()
