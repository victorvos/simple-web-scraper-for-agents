"""OpenAI-based AI agent for analyzing web content."""
import os
import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Union

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from app.agents.document_processor import DocumentProcessor
from app.agents.memory import AgentMemory, DEFAULT_MEMORY_PARAMS

# Configure logging
logger = logging.getLogger(__name__)

# Get environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
USE_FIREBASE_VECTORDB = os.environ.get("USE_FIREBASE_VECTORDB", "false").lower() == "true"


class OpenAIAgent:
    """Agent for analyzing web content using OpenAI."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        agent_id: Optional[str] = None,
        memory_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the OpenAI agent.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            agent_id: Unique identifier for this agent (auto-generated if not provided)
            memory_params: Custom memory parameters (uses defaults if not provided)
        """
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
            
        # Set or generate agent ID
        self.agent_id = agent_id or str(uuid.uuid4())
        
        # Initialize the language model
        self.llm = ChatOpenAI(
            openai_api_key=self.api_key,
            model="gpt-3.5-turbo-16k",
            temperature=0.2
        )
        
        # Text splitter for handling long documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=12000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize embeddings model
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        
        # Initialize document processor with same API key
        self.document_processor = DocumentProcessor(
            embeddings_model=self.embeddings
        )
        
        # Initialize agent memory
        self.memory = AgentMemory(
            agent_id=self.agent_id,
            embeddings_model=self.embeddings,
            use_firebase=USE_FIREBASE_VECTORDB,
            memory_params=memory_params
        )
        
        logger.info(f"Initialized OpenAI agent with ID: {self.agent_id}")
    
    async def get_memory_params(self) -> Dict[str, Any]:
        """
        Get current memory parameters.
        
        Returns:
            Dictionary with memory parameters
        """
        return await self.memory.get_memory_params()
    
    async def update_memory_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update memory parameters.
        
        Args:
            params: Dictionary with parameters to update
            
        Returns:
            Result dictionary
        """
        success = await self.memory.update_memory_params(params)
        
        if success:
            updated_params = await self.memory.get_memory_params()
            return {
                "success": True,
                "message": "Memory parameters updated successfully",
                "params": updated_params
            }
        else:
            return {
                "success": False,
                "error": "Failed to update memory parameters"
            }
    
    async def reset_memory_params(self) -> Dict[str, Any]:
        """
        Reset memory parameters to defaults.
        
        Returns:
            Result dictionary
        """
        success = await self.memory.update_memory_params(DEFAULT_MEMORY_PARAMS)
        
        if success:
            return {
                "success": True,
                "message": "Memory parameters reset to defaults",
                "params": DEFAULT_MEMORY_PARAMS
            }
        else:
            return {
                "success": False,
                "error": "Failed to reset memory parameters"
            }
