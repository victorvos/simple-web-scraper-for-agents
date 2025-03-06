"""Browser management for web scraping with anti-detection capabilities."""
import os
import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
import random
import time

# Configure logging
logger = logging.getLogger(__name__)

# Global browser instance
_browser: Optional[Browser] = None
_context: Optional[BrowserContext] = None
_browser_lock = asyncio.Lock()

# Get environment variables
HEADLESS = os.environ.get("BROWSER_HEADLESS", "true").lower() == "true"
USE_STEALTH_MODE = os.environ.get("USE_STEALTH_MODE", "true").lower() == "true"
SCRAPE_DELAY = float(os.environ.get("SCRAPE_DELAY", "2.0"))

# List of realistic user agents to rotate through
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0",
]

async def get_browser() -> Browser:
    """Get or launch a browser instance."""
    global _browser
    
    async with _browser_lock:
        if _browser is None or not _browser.is_connected():
            logger.info("Launching new browser instance")
            playwright = await async_playwright().start()
            _browser = await playwright.chromium.launch(
                headless=HEADLESS,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--disable-site-isolation-trials",
                ]
            )
    
    return _browser

async def get_stealth_context() -> BrowserContext:
    """Get a browser context with anti-fingerprinting features enabled."""
    global _context
    
    browser = await get_browser()
    
    async with _browser_lock:
        if _context is None or _context.browser != browser:
            logger.info("Creating new stealth browser context")
            
            # Random user agent
            user_agent = random.choice(USER_AGENTS)
            
            # Generate realistic viewport and screen size
            viewports = [(1280, 800), (1366, 768), (1440, 900), (1680, 1050), (1920, 1080)]
            width, height = random.choice(viewports)
            
            _context = await browser.new_context(
                user_agent=user_agent,
                viewport={"width": width, "height": height},
                screen={"width": width, "height": height},
                color_scheme="light",
                locale="en-US",
                timezone_id="Europe/Amsterdam",
                # Avoid sending webdriver flag
                bypass_csp=True,
                # Geolocation - Amsterdam by default
                geolocation={"latitude": 52.3676, "longitude": 4.9041},
                permissions=["geolocation"],
                # Extra options to prevent detection
                extra_http_headers={
                    "Accept-Language": "en-US,en;q=0.9,nl;q=0.8",
                },
                java_script_enabled=True,
            )

            # Execute additional scripts to evade detection
            if USE_STEALTH_MODE:
                await add_stealth_scripts(_context)
                
    return _context

async def add_stealth_scripts(context: BrowserContext) -> None:
    """Add scripts to evade browser fingerprinting detection."""
    # Create a page to execute scripts that will affect all future pages
    page = await context.new_page()
    
    try:
        # Remove webdriver properties
        await page.evaluate("""() => {
            // Overwrite the webdriver property
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false,
                configurable: true
            });
            
            // Remove automation-related properties
            delete navigator.__proto__.webdriver;
            
            // Modify chrome runtime to appear as a normal browser
            if (window.chrome) {
                window.chrome.runtime = {};
            }
            
            // Add language plugins that a normal browser would have
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en', 'nl'],
                configurable: true
            });
        }""")
        
        # Add permissions a normal browser would have
        await page.evaluate("""() => {
            // Mock permissions API
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
        }""")
        
        # Add plugins that a normal browser would have
        await page.evaluate("""() => {
            // Mock plugin array to appear like a normal browser
            Object.defineProperty(navigator, 'plugins', {
                get: () => {
                    return [
                        { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer' },
                        { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai' },
                        { name: 'Native Client', filename: 'internal-nacl-plugin' }
                    ];
                }
            });
        }""")
        
        logger.info("Stealth scripts added to browser context")
    finally:
        await page.close()

async def get_page() -> Page:
    """Get a page with stealth features enabled."""
    context = await get_stealth_context()
    page = await context.new_page()
    
    # Add a random delay to look more like a human user
    delay = SCRAPE_DELAY * (0.8 + random.random() * 0.4)  # 80-120% of configured delay
    await asyncio.sleep(delay)
    
    return page

async def close_browser() -> None:
    """Close the browser instance."""
    global _browser, _context
    
    async with _browser_lock:
        if _context:
            await _context.close()
            _context = None
            
        if _browser:
            await _browser.close()
            _browser = None
            
    logger.info("Browser resources released")
