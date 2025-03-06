#!/usr/bin/env python
"""Test script for Firebase vector database integration."""
import os
import sys
import asyncio
import logging
import argparse
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import app modules
from app.scraper.scraper import WebScraper
from app.storage.firebase_vector_db import FirebaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def test_firebase_vector_db(url: str, query: str):
    """
    Test Firebase vector database by scraping a URL, vectorizing the content, 
    and performing a similarity search.
    
    Args:
        url: URL to scrape
        query: Query to search for
    """
    try:
        # Check if Firebase is configured
        firebase_credentials_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")
        if not firebase_credentials_path or not os.path.exists(firebase_credentials_path):
            logger.error(f"Firebase credentials not found at {firebase_credentials_path}")
            logger.error("Please set FIREBASE_CREDENTIALS_PATH in your .env file")
            return
        
        # 1. Scrape URL
        logger.info(f"Scraping {url}...")
        result = await WebScraper.scrape_url(url=url)
        
        if not result.get("success", False):
            logger.error(f"Failed to scrape URL: {result.get('error', 'Unknown error')}")
            return
        
        # 2. Extract text data
        text = result.get("text", "")
        title = result.get("title", "")
        
        logger.info(f"Successfully scraped {len(text)} characters from {url}")
        logger.info(f"Title: {title}")
        
        # 3. Initialize Firebase vector store
        logger.info("Initializing Firebase vector store...")
        
        embeddings = OpenAIEmbeddings()
        vector_store = FirebaseVectorStore(
            embedding_model=embeddings,
            collection_name=os.environ.get("FIREBASE_COLLECTION_NAME", "vector_embeddings"),
            namespace=os.environ.get("FIREBASE_NAMESPACE", "web_scraper"),
            index_name=f"test-{hash(url) % 10000}"  # Use a deterministic index name for testing
        )
        
        # 4. Create a test document
        logger.info("Creating test documents...")
        
        # Create a short sample document so we don't use too many tokens
        sample_text = text[:2000] if len(text) > 2000 else text
        
        documents = [
            Document(
                page_content=sample_text,
                metadata={
                    "url": url,
                    "title": title,
                    "test": True
                }
            )
        ]
        
        # 5. Add documents to vector store
        vector_store.add_documents(documents)
        logger.info("Added documents to Firebase vector store")
        
        # 6. Test similarity search
        logger.info(f"Testing similarity search with query: '{query}'")
        results = vector_store.similarity_search(query, k=1)
        
        if results:
            logger.info("Similarity search successful!")
            logger.info(f"Found {len(results)} matching documents")
            logger.info("First result snippet: " + results[0].page_content[:100] + "...")
        else:
            logger.warning("No results found for similarity search")
        
        # 7. Test deletion
        logger.info("Testing vector deletion...")
        vector_store.delete()
        logger.info("Successfully deleted test vectors from Firebase")
        
        logger.info("Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing Firebase vector database: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Firebase vector database integration")
    parser.add_argument("url", help="URL to scrape and vectorize")
    parser.add_argument("query", help="Query to search for in the vectorized content")
    args = parser.parse_args()
    
    # Run the test
    asyncio.run(test_firebase_vector_db(args.url, args.query))

if __name__ == "__main__":
    main()
