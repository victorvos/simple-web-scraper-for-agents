"""
Test script for analyzing web content with OpenAI.

Usage:
    python -m scripts.test_analyze [url] [question] [--no-cache] [--no-vector]

Examples:
    python -m scripts.test_analyze https://en.wikipedia.org/wiki/Web_scraping "What are the ethical concerns with web scraping?"
    python -m scripts.test_analyze https://news.ycombinator.com "What are the trending topics on this page?" --no-vector
"""

import asyncio
import argparse
import json
import os
import sys
from pprint import pprint
from dotenv import load_dotenv
import time

# Add the parent directory to PATH to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the required modules
from app.scraper.scraper import WebScraper
from app.agents.openai_agent import OpenAIAgent

# Load environment variables
load_dotenv()

async def test_analyze(url, question, use_cache=True, use_vectorization=True):
    """Test the AI analysis with a given URL and question."""
    print(f"Analyzing URL: {url}")
    print(f"Question: {question}")
    print(f"Using cache: {use_cache}")
    print(f"Using vectorization: {use_vectorization}")
    
    try:
        # First, scrape the URL
        print("\nScraping content...")
        start_time = time.time()
        content = await WebScraper.scrape_url(
            url=url,
            use_cache=use_cache
        )
        scrape_time = time.time() - start_time
        
        if not content.get("success", False):
            print("Scraping failed:")
            print(content.get("error", "Unknown error"))
            return
            
        word_count = len(content.get('text', '').split())
        char_count = len(content.get('text', ''))
        estimated_tokens = char_count // 4  # Rough approximation: 4 chars ≈ 1 token
        
        print(f"Successfully scraped {word_count} words ({char_count} chars) of content in {scrape_time:.2f} seconds")
        print(f"Estimated tokens in full text: ~{estimated_tokens} (${estimated_tokens * 0.0002:.4f} with GPT-4)")
        
        # Then, analyze with OpenAI
        print("\nAnalyzing with OpenAI...")
        try:
            openai_agent = OpenAIAgent()
        except ValueError as e:
            print(f"Error initializing OpenAI agent: {e}")
            print("Make sure your OpenAI API key is set in the .env file")
            return
            
        # Measure the analysis time
        start_time = time.time()
        analysis = await openai_agent.analyze_content(
            content=content,
            question=question,
            use_vectorization=use_vectorization
        )
        analysis_time = time.time() - start_time
        
        # Print results
        if analysis.get("success", False):
            print("\n--- Analysis Results ---")
            print(f"Question: {question}")
            print(f"\nAnswer:\n{analysis.get('answer', 'No answer provided')}")
            
            # Print vectorization stats if available
            if "vectorization" in analysis and use_vectorization:
                v_stats = analysis["vectorization"]
                original_length = v_stats['original_length']
                optimized_length = v_stats['optimized_length']
                compression_ratio = v_stats['compression_ratio']
                tokens_saved = (original_length - optimized_length) // 4
                cost_saved = tokens_saved * 0.0002  # Approximate cost for GPT-4 tokens
                
                print("\n--- Vectorization Stats ---")
                print(f"Original length: {original_length:,} chars (~{original_length//4:,} tokens)")
                print(f"Optimized length: {optimized_length:,} chars (~{optimized_length//4:,} tokens)")
                print(f"Compression ratio: {compression_ratio * 100:.1f}% of original size")
                print(f"Tokens saved: ~{tokens_saved:,} (${cost_saved:.4f} with GPT-4)")
                print(f"Analysis completed in {analysis_time:.2f} seconds")
                
                print("\n--- Benefits of Vectorization ---")
                print("1. Cost efficiency: Only the most relevant content is sent to the LLM")
                print("2. Better responses: LLMs perform better with focused, relevant context")
                print("3. Larger documents: Can handle documents far beyond context limits")
                print("4. Speed: Fewer tokens means faster processing and responses")
                
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
    parser.add_argument("--no-vector", action="store_true", help="Disable vectorization")
    
    args = parser.parse_args()
    
    # Run the test
    asyncio.run(test_analyze(
        url=args.url,
        question=args.question,
        use_cache=not args.no_cache,
        use_vectorization=not args.no_vector
    ))

if __name__ == "__main__":
    main()
