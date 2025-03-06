#!/usr/bin/env python
"""Test script for agent memory parameters functionality."""
import os
import sys
import asyncio
import logging
import argparse
import json
from pprint import pprint
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import app modules
from app.agents.openai_agent import OpenAIAgent
from app.agents.memory import DEFAULT_MEMORY_PARAMS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def test_memory_parameters():
    """Test memory parameters management."""
    try:
        # 1. Initialize agent with default parameters
        agent_id = "test-params-agent"
        agent = OpenAIAgent(agent_id=agent_id)
        logger.info(f"Initialized agent with ID: {agent_id}")
        
        # 2. Get default parameters
        default_params = await agent.get_memory_params()
        logger.info("Default memory parameters:")
        pprint(default_params)
        
        # 3. Update some parameters
        custom_params = {
            "max_items_per_query": 10,
            "relevance_threshold": 0.7,
            "priority_categories": ["important", "critical"],
            "context_strategy": "priority_first"
        }
        
        update_result = await agent.update_memory_params(custom_params)
        
        if update_result.get("success", False):
            logger.info("Parameters updated successfully:")
            pprint(update_result.get("params", {}))
        else:
            logger.error(f"Failed to update parameters: {update_result.get('error')}")
            return False
        
        # 4. Verify the updates
        updated_params = await agent.get_memory_params()
        
        for key, value in custom_params.items():
            if updated_params.get(key) != value:
                logger.error(f"Parameter {key} was not updated correctly")
                return False
                
        logger.info("All parameters were updated correctly")
        
        # 5. Reset parameters to defaults
        reset_result = await agent.reset_memory_params()
        
        if reset_result.get("success", False):
            logger.info("Parameters reset successfully:")
            pprint(reset_result.get("params", {}))
        else:
            logger.error(f"Failed to reset parameters: {reset_result.get('error')}")
            return False
        
        # 6. Verify the reset
        reset_params = await agent.get_memory_params()
        
        for key, value in DEFAULT_MEMORY_PARAMS.items():
            if reset_params.get(key) != value:
                logger.error(f"Parameter {key} was not reset correctly")
                return False
                
        logger.info("All parameters were reset correctly")
        
        # 7. Test parameter persistence
        # Create a new agent with the same ID to test if parameters persist
        new_agent = OpenAIAgent(agent_id=agent_id)
        persisted_params = await new_agent.get_memory_params()
        
        for key, value in DEFAULT_MEMORY_PARAMS.items():
            if persisted_params.get(key) != value:
                logger.error(f"Parameter {key} did not persist correctly")
                return False
                
        logger.info("All parameters persisted correctly between agent instances")
        
        # 8. Test updating partial parameters
        partial_update = {
            "max_context_items": 7,
            "auto_categorize": False
        }
        
        await new_agent.update_memory_params(partial_update)
        partial_result = await new_agent.get_memory_params()
        
        for key, value in partial_update.items():
            if partial_result.get(key) != value:
                logger.error(f"Partial update of parameter {key} failed")
                return False
                
        # Check that other parameters remained unchanged
        for key, value in DEFAULT_MEMORY_PARAMS.items():
            if key not in partial_update and partial_result.get(key) != value:
                logger.error(f"Parameter {key} was changed during partial update")
                return False
                
        logger.info("Partial parameter update worked correctly")
        
        # 9. Final reset
        await new_agent.reset_memory_params()
        
        return True
        
    except Exception as e:
        logger.error(f"Error in test_memory_parameters: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test agent memory parameters")
    args = parser.parse_args()
    
    result = asyncio.run(test_memory_parameters())
    
    if result:
        logger.info("Memory parameters test completed successfully!")
        sys.exit(0)
    else:
        logger.error("Memory parameters test failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
