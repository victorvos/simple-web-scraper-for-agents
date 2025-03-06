"""Core web scraping functionality."""
import os
import asyncio
import logging
import re
import random
import time
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from app.scraper.browser import get_page
from app.storage.cache import save_to_cache, load_from_cache, get_cache_key

# Configure logging
logger = logging.getLogger(__name__)

# Get environment variables
MAX_PAGES = int(os.environ.get("MAX_PAGES", "3"))


class WebScraper:
    """Web scraper with anti-detection measures."""
    
    @staticmethod
    async def scrape_url(
        url: str, 
        selector: Optional[str] = None,
        use_cache: bool = True,
        scroll: bool = True,
        wait_for: Optional[str] = None,
        extract_links: bool = False,
        max_wait_time: int = 10000
    ) -> Dict[str, Any]:
        """
        Scrape content from a URL.
        
        Args:
            url: URL to scrape
            selector: CSS selector to extract specific content (if None, extracts everything)
            use_cache: Whether to use cached results if available
            scroll: Whether to scroll the page to load lazy content
            wait_for: Additional selector to wait for before extracting
            extract_links: Whether to extract links from the page
            max_wait_time: Maximum time to wait for selectors (ms)
            
        Returns:
            Dictionary with scraped content and metadata
        """
        # Check cache first if enabled
        if use_cache:
            cache_key = get_cache_key(url, selector)
            cached_data = await load_from_cache(cache_key)
            if cached_data:
                logger.info(f"Using cached content for {url}")
                return cached_data
        
        logger.info(f"Scraping URL: {url}")
        page = None
        
        try:
            # Get a browser page with stealth features
            page = await get_page()
            
            # Add random delays between actions to appear more human-like
            await asyncio.sleep(random.uniform(1, 3))
            
            # Navigate to the URL with a timeout
            response = await page.goto(url, wait_until="domcontentloaded", timeout=max_wait_time)
            
            # Check if the page loaded successfully
            if not response or response.status >= 400:
                return {
                    "success": False,
                    "error": f"Failed to load page: HTTP {response.status if response else 'Unknown'}"
                }
            
            # Wait for any specified selector
            if wait_for:
                try:
                    await page.wait_for_selector(wait_for, timeout=max_wait_time)
                except PlaywrightTimeoutError:
                    logger.warning(f"Timeout waiting for selector '{wait_for}' on {url}")
            
            # If no specific selector, wait for the body to be loaded
            else:
                await page.wait_for_selector("body", timeout=max_wait_time)
            
            # Scroll to load lazy content if requested
            if scroll:
                await WebScraper._scroll_page(page)
            
            # Wait a moment for any lazy-loaded content to appear
            await asyncio.sleep(random.uniform(1, 2))
            
            # Extract page title
            title = await page.title()
            
            # Extract page content
            if selector:
                try:
                    # Wait for specific content
                    await page.wait_for_selector(selector, timeout=max_wait_time)
                    content_html = await page.inner_html(selector)
                except PlaywrightTimeoutError:
                    logger.warning(f"Selector '{selector}' not found, falling back to body")
                    content_html = await page.content()
            else:
                content_html = await page.content()
            
            # Parse with BeautifulSoup for easier text extraction
            soup = BeautifulSoup(content_html, "html.parser")
            
            # Extract text content
            text_content = WebScraper._extract_text(soup)
            
            # Extract links if requested
            links = []
            if extract_links:
                links = await WebScraper._extract_links(page, url)
            
            # Create result
            result = {
                "success": True,
                "url": url,
                "title": title,
                "html": content_html,
                "text": text_content,
                "links": links,
                "timestamp": time.time()
            }
            
            # Save to cache if enabled
            if use_cache:
                await save_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }
            
        finally:
            if page:
                await page.close()
    
    @staticmethod
    async def _scroll_page(page: Page) -> None:
        """Scroll the page to load lazy content in a human-like way."""
        # Get page height
        page_height = await page.evaluate("document.body.scrollHeight")
        viewport_height = await page.evaluate("window.innerHeight")
        
        if not page_height or not viewport_height:
            return
        
        # Calculate number of scrolls needed with some overlap
        num_scrolls = max(1, round(page_height / (viewport_height * 0.8)))
        
        # Scroll in chunks with random delays
        for i in range(num_scrolls):
            # Random scroll position with some variation
            position = min(page_height, (i + 1) * viewport_height * 0.8)
            position_with_jitter = position * (0.95 + random.random() * 0.1)  # Add 5% jitter
            
            # Scroll smoothly using mouse wheel simulation
            await page.evaluate(f"window.scrollTo({{top: {position_with_jitter}, behavior: 'smooth'}})")
            
            # Random pause like a human would
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            # Sometimes move the mouse to appear more human-like
            if random.random() < 0.3:  # 30% chance
                x = random.randint(100, 700)
                y = random.randint(100, 500)
                await page.mouse.move(x, y)
    
    @staticmethod
    def _extract_text(soup: BeautifulSoup) -> str:
        """Extract readable text content from HTML."""
        # Remove script and style elements
        for element in soup(["script", "style", "noscript", "iframe", "svg"]):
            element.extract()
        
        # Get text
        text = soup.get_text(separator="\n", strip=True)
        
        # Remove excessive newlines and whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    @staticmethod
    async def _extract_links(page: Page, base_url: str) -> List[Dict[str, str]]:
        """Extract links from the page with their text."""
        # Extract all links using JavaScript for better accuracy
        links_data = await page.evaluate("""() => {
            const links = Array.from(document.querySelectorAll('a[href]'));
            return links.map(link => {
                return {
                    href: link.href,
                    text: link.innerText.trim(),
                    title: link.getAttribute('title') || ''
                };
            });
        }""")
        
        # Filter and normalize links
        result = []
        seen_urls = set()
        parsed_base = urlparse(base_url)
        
        for link in links_data:
            href = link.get("href", "")
            
            # Skip empty links and anchors
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue
            
            # Normalize URL
            href = urljoin(base_url, href)
            
            # Skip duplicates
            if href in seen_urls:
                continue
                
            seen_urls.add(href)
            
            # Only include links from the same domain
            parsed_href = urlparse(href)
            is_same_domain = parsed_href.netloc == parsed_base.netloc
            
            result.append({
                "url": href,
                "text": link.get("text", "").strip(),
                "title": link.get("title", "").strip(),
                "same_domain": is_same_domain
            })
        
        return result

    @staticmethod
    async def scrape_multiple_pages(
        start_url: str,
        max_pages: int = MAX_PAGES,
        same_domain_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Scrape multiple pages starting from a URL and following links.
        
        Args:
            start_url: Starting URL to scrape
            max_pages: Maximum number of pages to scrape
            same_domain_only: Whether to only follow links on the same domain
            
        Returns:
            List of dictionaries with scraped content from each page
        """
        pages_to_visit = [start_url]
        visited_pages = set()
        results = []
        
        while pages_to_visit and len(visited_pages) < max_pages:
            # Get next URL to visit
            current_url = pages_to_visit.pop(0)
            
            # Skip if already visited
            if current_url in visited_pages:
                continue
                
            # Mark as visited
            visited_pages.add(current_url)
            
            # Scrape the page
            result = await WebScraper.scrape_url(
                current_url, 
                extract_links=True, 
                use_cache=True
            )
            
            # Add to results if successful
            if result.get("success", False):
                results.append(result)
                
                # Add links to visit
                links = result.get("links", [])
                for link in links:
                    url = link.get("url")
                    if url and url not in visited_pages and url not in pages_to_visit:
                        # Only follow links on the same domain if specified
                        if not same_domain_only or link.get("same_domain", False):
                            pages_to_visit.append(url)
            
            # Random delay between pages
            await asyncio.sleep(random.uniform(2, 5))
        
        return results
