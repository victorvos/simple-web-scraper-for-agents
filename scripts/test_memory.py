#!/usr/bin/env python
"""Test script for agent memory functionality."""
import os
import sys
import asyncio
import logging
import argparse
import json
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import app modules
from app.agents.openai_agent import OpenAIAgent
from app.scraper.scraper import WebScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def test_add_memory(agent: OpenAIAgent, content: str, category: str = "test") -> str:
    """
    Test adding content to agent memory.
    
    Args:
        agent: OpenAI agent instance
        content: Content to add
        category: Memory category
        
    Returns:
        Memory ID
    """
    logger.info(f"Adding test content to memory, category: {category}")
    
    result = await agent.add_to_memory(
        content=content,
        content_type="document",
        category=category,
        metadata={"test": True, "source": "test_script"}
    )
    
    if result.get("success", False):
        memory_id = result.get("memory_id")
        logger.info(f"Successfully added content to memory with ID: {memory_id}")
        return memory_id
    else:
        error = result.get("error", "Unknown error")
        logger.error(f"Failed to add content to memory: {error}")
        raise Exception(f"Failed to add content to memory: {error}")

async def test_add_data_object(agent: OpenAIAgent, data: Dict[str, Any], description: str, category: str = "test_data") -> str:
    """
    Test adding a data object to agent memory.
    
    Args:
        agent: OpenAI agent instance
        data: Data object
        description: Description of the data
        category: Memory category
        
    Returns:
        Memory ID
    """
    logger.info(f"Adding test data object to memory, category: {category}")
    
    result = await agent.add_to_memory(
        content=data,
        content_type="data_object",
        description=description,
        category=category
    )
    
    if result.get("success", False):
        memory_id = result.get("memory_id")
        logger.info(f"Successfully added data object to memory with ID: {memory_id}")
        return memory_id
    else:
        error = result.get("error", "Unknown error")
        logger.error(f"Failed to add data object to memory: {error}")
        raise Exception(f"Failed to add data object to memory: {error}")

async def test_retrieve_memory(agent: OpenAIAgent, query: str) -> List[Dict[str, Any]]:
    """
    Test retrieving memory by query.
    
    Args:
        agent: OpenAI agent instance
        query: Query to search for
        
    Returns:
        List of memory items
    """
    logger.info(f"Retrieving memory with query: {query}")
    
    result = await agent.get_memory(query=query)
    
    if result.get("success", False):
        memories = result.get("memories", [])
        count = len(memories)
        logger.info(f"Retrieved {count} memory items")
        return memories
    else:
        error = result.get("error", "Unknown error")
        logger.error(f"Failed to retrieve memory: {error}")
        return []

async def test_get_by_id(agent: OpenAIAgent, memory_id: str) -> Dict[str, Any]:
    """
    Test retrieving memory by ID.
    
    Args:
        agent: OpenAI agent instance
        memory_id: Memory ID
        
    Returns:
        Memory item
    """
    logger.info(f"Getting memory with ID: {memory_id}")
    
    result = await agent.get_memory(memory_id=memory_id)
    
    if result.get("success", False):
        memory = result.get("memory")
        logger.info(f"Successfully retrieved memory: {memory_id}")
        return memory
    else:
        error = result.get("error", "Unknown error")
        logger.error(f"Failed to get memory: {error}")
        return {}

async def test_list_by_category(agent: OpenAIAgent, category: str) -> List[Dict[str, Any]]:
    """
    Test listing memory by category.
    
    Args:
        agent: OpenAI agent instance
        category: Category to list
        
    Returns:
        List of memory items
    """
    logger.info(f"Listing memory in category: {category}")
    
    result = await agent.get_memory(category=category)
    
    if result.get("success", False):
        memories = result.get("memories", [])
        count = len(memories)
        logger.info(f"Listed {count} memory items in category: {category}")
        return memories
    else:
        error = result.get("error", "Unknown error")
        logger.error(f"Failed to list memory by category: {error}")
        return []

async def test_delete_memory(agent: OpenAIAgent, memory_id: str, memory_type: str = "document") -> bool:
    """
    Test deleting memory.
    
    Args:
        agent: OpenAI agent instance
        memory_id: Memory ID to delete
        memory_type: Type of memory
        
    Returns:
        True if successful
    """
    logger.info(f"Deleting memory with ID: {memory_id}")
    
    result = await agent.delete_memory(memory_id, memory_type)
    
    if result.get("success", False):
        logger.info(f"Successfully deleted memory: {memory_id}")
        return True
    else:
        error = result.get("error", "Unknown error")
        logger.error(f"Failed to delete memory: {error}")
        return False

async def test_analyze_with_memory(agent: OpenAIAgent, url: str, question: str) -> Dict[str, Any]:
    """
    Test analyzing content with memory.
    
    Args:
        agent: OpenAI agent instance
        url: URL to analyze
        question: Question to answer
        
    Returns:
        Analysis result
    """
    logger.info(f"Scraping URL: {url}")
    
    # Scrape the URL
    content = await WebScraper.scrape_url(url=url)
    
    if not content.get("success", False):
        error = content.get("error", "Unknown error")
        logger.error(f"Failed to scrape URL: {error}")
        return {"success": False, "error": error}
    
    logger.info(f"Analyzing content with memory integration")
    
    # Analyze with memory integration
    result = await agent.analyze_content(
        content=content,
        question=question,
        use_vectorization=True,
        use_memory=True
    )
    
    if result.get("success", False):
        answer = result.get("answer", "")
        memory_used = result.get("memory_used", False)
        logger.info(f"Successfully analyzed content, memory used: {memory_used}")
        logger.info(f"Answer snippet: {answer[:100]}...")
        return result
    else:
        error = result.get("error", "Unknown error")
        logger.error(f"Failed to analyze content: {error}")
        return {"success": False, "error": error}

async def test_memory_integration():
    """
    Test the full memory integration.
    """
    try:
        # Initialize agent
        agent_id = "test-agent-" + os.urandom(4).hex()
        agent = OpenAIAgent(agent_id=agent_id)
        
        logger.info(f"Testing memory with agent ID: {agent_id}")
        
        # Step 1: Add some test content to memory
        content1 = """Artificial intelligence (AI) is intelligence demonstrated by machines, 
            as opposed to intelligence displayed by humans or other animals. Example tasks in 
            which this is done include speech recognition, computer vision, translation between 
            languages, and decision making."""
        memory_id1 = await test_add_memory(agent, content1, "AI")
        
        content2 = """Machine learning (ML) is a type of artificial intelligence (AI) that 
            allows software applications to become more accurate at predicting outcomes without 
            being explicitly programmed to do so. Machine learning algorithms use historical 
            data as input to predict new output values."""
        memory_id2 = await test_add_memory(agent, content2, "AI")
        
        # Step 2: Add a data object
        data_obj = {
            "name": "AI Technologies",
            "categories": ["Machine Learning", "Neural Networks", "NLP"],
            "popularity": {
                "ML": 85,
                "Neural Networks": 78,
                "NLP": 72
            }
        }
        data_id = await test_add_data_object(
            agent,
            data_obj,
            "Statistical data about AI technology popularity in 2023",
            "AI_stats"
        )
        
        # Step 3: Retrieve by query
        memories = await test_retrieve_memory(agent, "What is artificial intelligence?")
        
        # Step 4: Get by ID
        memory = await test_get_by_id(agent, memory_id1)
        
        # Step 5: List by category
        category_items = await test_list_by_category(agent, "AI")
        
        # Step 6: Analyze content with memory
        analysis = await test_analyze_with_memory(
            agent,
            "https://en.wikipedia.org/wiki/Deep_learning",
            "How does deep learning relate to artificial intelligence and machine learning?"
        )
        
        # Step 7: Test with a new question that should use memory
        analysis2 = await test_analyze_with_memory(
            agent,
            "https://en.wikipedia.org/wiki/Convolutional_neural_network",
            "What are the popularity statistics of different AI technologies?"
        )
        
        # Step 8: Delete one memory item
        deleted = await test_delete_memory(agent, memory_id2)
        
        # Step 9: Verify deletion by retrieving again
        memories_after = await test_retrieve_memory(agent, "machine learning")
        
        # Step 10: Success message
        logger.info("Memory integration test completed successfully")
        
        return {
            "success": True,
            "agent_id": agent_id,
            "memory_counts": {
                "initial_query": len(memories),
                "after_deletion": len(memories_after)
            },
            "analyses": {
                "memory_used1": analysis.get("memory_used", False),
                "memory_used2": analysis2.get("memory_used", False)
            }
        }
        
    except Exception as e:
        logger.error(f"Memory integration test failed: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test agent memory functionality")
    parser.add_argument("--simple", action="store_true", help="Run simplified memory test")
    parser.add_argument("--url", help="URL to analyze (for simple test)")
    parser.add_argument("--question", help="Question to answer (for simple test)")
    args = parser.parse_args()
    
    if args.simple and args.url and args.question:
        # Run simple test
        async def simple_test():
            agent = OpenAIAgent()
            logger.info(f"Agent ID: {agent.agent_id}")
            
            # Add test memory
            await test_add_memory(
                agent,
                "LLMs are large language models that can generate text based on patterns learned from training data.",
                "AI"
            )
            
            # Analyze with memory
            result = await test_analyze_with_memory(agent, args.url, args.question)
            
            # Format result
            if result.get("success", False):
                logger.info("\n" + "="*80)
                logger.info("QUESTION: " + args.question)
                logger.info("ANSWER: " + result.get("answer", "No answer"))
                logger.info("Memory used: " + str(result.get("memory_used", False)))
                logger.info("="*80)
                
        asyncio.run(simple_test())
    else:
        # Run full integration test
        result = asyncio.run(test_memory_integration())
        if result.get("success", False):
            logger.info("All memory tests passed successfully!")
        else:
            logger.error(f"Memory tests failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)

if __name__ == "__main__":
    main()
