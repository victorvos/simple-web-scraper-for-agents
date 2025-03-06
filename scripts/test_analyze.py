"""
Test script for analyzing web content with OpenAI.

Usage:
    python -m scripts.test_analyze [url] [question] [--no-cache]

Examples:
    python -m scripts.test_analyze https://en.wikipedia.org/wiki/Web_scraping "What are the ethical concerns with web scraping?"
    python -m scripts.test_analyze https://news.ycombinator.com "What are the trending topics on this page?"
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

# Import the required modules
from app.scraper.scraper import WebScraper
from app.agents.openai_agent import OpenAIAgent

# Load environment variables
load_dotenv()

async def test_analyze(url, question, use_cache=True):
    """Test the AI analysis with a given URL and question."""
    print(f"Analyzing URL: {url}")
    print(f"Question: {question}")
    print(f"Using cache: {use_cache}")
    
    try:
        # First, scrape the URL
        print("\nScraping content...")
        content = await WebScraper.scrape_url(
            url=url,
            use_cache=use_cache
        )
        
        if not content.get("success", False):
            print("Scraping failed:")
            print(content.get("error", "Unknown error"))
            return
            
        print(f"Successfully scraped {len(content.get('text', '').split())} words of content")
        
        # Then, analyze with OpenAI
        print("\nAnalyzing with OpenAI...")
        try:
            openai_agent = OpenAIAgent()
        except ValueError as e:
            print(f"Error initializing OpenAI agent: {e}")
            print("Make sure your OpenAI API key is set in the .env file")
            return
            
        analysis = await openai_agent.analyze_content(
            content=content,
            question=question
        )
        
        # Print results
        if analysis.get("success", False):
            print("\n--- Analysis Results ---")
            print(f"Question: {question}")
            print(f"\nAnswer:\n{analysis.get('answer', 'No answer provided')}")
                
            # Save results to file
            output_file = "analysis_result.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
                
            print(f"\nFull results saved to {output_file}")
            
        else:
            print("\nAnalysis failed:")
            print(analysis.get("error", "Unknown error"))
            
    except Exception as e:
        print(f"Error: {str(e)}")
        
def main():
    """Parse arguments and run test."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test AI analysis of web content")
    parser.add_argument("url", help="URL to analyze")
    parser.add_argument("question", help="Question to answer about the content")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache")
    
    args = parser.parse_args()
    
    # Run the test
    asyncio.run(test_analyze(
        url=args.url,
        question=args.question,
        use_cache=not args.no_cache
    ))

if __name__ == "__main__":
    main()
