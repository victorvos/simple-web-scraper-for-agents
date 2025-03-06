"""
Test script for the web scraper with a specific website.

Usage:
    python -m scripts.test_scrape [url] [--selector=main] [--no-cache]

Examples:
    python -m scripts.test_scrape https://en.wikipedia.org/wiki/Web_scraping
    python -m scripts.test_scrape https://news.ycombinator.com --selector=".storylink"
"""

import asyncio
import argparse
import json
import os
import sys
from pprint import pprint
from dotenv import load_dotenv

# Add the parent directory to PATH to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the scraper
from app.scraper.scraper import WebScraper

# Load environment variables
load_dotenv()

async def test_scrape(url, selector=None, use_cache=True):
    """Test the scraper with a given URL and selector."""
    print(f"Scraping URL: {url}")
    print(f"Selector: {selector or 'None'}")
    print(f"Using cache: {use_cache}")
    
    try:
        # Scrape the URL
        result = await WebScraper.scrape_url(
            url=url,
            selector=selector,
            use_cache=use_cache,
            extract_links=True,
            scroll=True
        )
        
        # Print results
        if result.get("success", False):
            print("\n--- Scraping Results ---")
            print(f"Title: {result.get('title', 'N/A')}")
            
            # Print text preview
            text = result.get("text", "")
            text_preview = text[:500] + "..." if len(text) > 500 else text
            print(f"\nText Preview:\n{text_preview}")
            
            # Print HTML snippet
            html = result.get("html", "")
            html_preview = html[:200] + "..." if len(html) > 200 else html
            print(f"\nHTML Snippet:\n{html_preview}")
            
            # Print links
            links = result.get("links", [])
            if links:
                print(f"\nFound {len(links)} links")
                print("First 5 links:")
                for i, link in enumerate(links[:5]):
                    print(f"  {i+1}. {link.get('text', 'N/A')[:30]}: {link.get('url', 'N/A')}")
                    
            # Save results to file
            output_file = "scrape_result.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            print(f"\nFull results saved to {output_file}")
            
        else:
            print("\nScraping failed:")
            print(result.get("error", "Unknown error"))
            
    except Exception as e:
        print(f"Error: {str(e)}")
        
def main():
    """Parse arguments and run test."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test the web scraper")
    parser.add_argument("url", help="URL to scrape")
    parser.add_argument("--selector", help="CSS selector to extract (e.g., 'main', '.content')")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache")
    
    args = parser.parse_args()
    
    # Run the test
    asyncio.run(test_scrape(
        url=args.url,
        selector=args.selector,
        use_cache=not args.no_cache
    ))

if __name__ == "__main__":
    main()
