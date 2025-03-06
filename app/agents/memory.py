"""Memory management for agents using vector database storage."""
import os
import logging
import hashlib
import json
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from uuid import uuid4

from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

# Import Firebase vector store
from app.storage.firebase_vector_db import FirebaseVectorStore
from langchain.vectorstores import FAISS

# Configure logging
logger = logging.getLogger(__name__)

# Get environment variables
USE_FIREBASE_VECTORDB = os.environ.get("USE_FIREBASE_VECTORDB", "false").lower() == "true"
FIREBASE_COLLECTION_NAME = os.environ.get("FIREBASE_COLLECTION_NAME", "vector_embeddings")
FIREBASE_NAMESPACE = os.environ.get("FIREBASE_NAMESPACE", "web_scraper")
MEMORY_CACHE_DIR = os.environ.get("MEMORY_CACHE_DIR", "./data/memory_cache")

# Default memory parameters
DEFAULT_MEMORY_PARAMS = {
    "max_items_per_query": 5,       # Maximum number of memory items to retrieve per query
    "relevance_threshold": 0.6,     # Minimum similarity score (0-1) for memory to be considered relevant
    "max_context_items": 3,         # Maximum number of memory items to include in context
    "retention_period": 2592000,    # How long to retain memories in seconds (30 days default)
    "priority_categories": [],      # Categories that should be prioritized in retrieval
    "auto_categorize": True,        # Whether to attempt automatic categorization of new memories
    "context_strategy": "recency_weighted"  # Strategy for context selection (options: relevance_only, recency_weighted, priority_first)
}

class AgentMemory:
    """
    Context memory for agents using vector database storage.
    
    Features:
    - Store documents and data objects with metadata
    - Retrieve relevant context based on queries
    - Maintain persistent memory across sessions
    - Tag memory objects with categories
    - Filter memory by metadata
    - Configurable memory parameters
    """
    
    def __init__(
        self,
        agent_id: str,
        embeddings_model: Optional[Embeddings] = None,
        use_firebase: bool = USE_FIREBASE_VECTORDB,
        firebase_collection: str = FIREBASE_COLLECTION_NAME,
        firebase_namespace: str = FIREBASE_NAMESPACE,
        memory_cache_dir: str = MEMORY_CACHE_DIR,
        memory_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize agent memory.
        
        Args:
            agent_id: Unique identifier for the agent
            embeddings_model: Model to use for embeddings
            use_firebase: Whether to use Firebase for storage
            firebase_collection: Firestore collection name
            firebase_namespace: Firebase namespace
            memory_cache_dir: Directory for local cache
            memory_params: Optional custom memory parameters
        """
        self.agent_id = agent_id
        self.embeddings = embeddings_model or OpenAIEmbeddings()
        self.use_firebase = use_firebase
        self.firebase_collection = firebase_collection
        self.firebase_namespace = firebase_namespace
        self.memory_cache_dir = memory_cache_dir
        
        # Set memory parameters (use defaults for any missing)
        self.memory_params = DEFAULT_MEMORY_PARAMS.copy()
        if memory_params:
            self.memory_params.update(memory_params)
        
        # Initialize memory store
        self._initialize_memory_store()
        
        # Load or create agent parameters
        self._load_agent_params()
        
        logger.info(f"Initialized memory for agent {agent_id}")
        logger.info(f"Using Firebase: {use_firebase}")
    
    def _initialize_memory_store(self):
        """Initialize the vector store for memory."""
        # Create namespace for this agent's memory
        memory_namespace = f"{self.firebase_namespace}_memory_{self.agent_id}"
        
        # Initialize the vector store
        if self.use_firebase:
            try:
                # Use Firebase
                self.memory_store = FirebaseVectorStore(
                    embedding_model=self.embeddings,
                    collection_name=self.firebase_collection,
                    namespace=memory_namespace,
                    index_name="agent_memory"
                )
                logger.info(f"Using Firebase for agent memory: {memory_namespace}")
            except Exception as e:
                logger.error(f"Failed to initialize Firebase memory: {str(e)}")
                logger.info("Falling back to local FAISS memory store")
                self.use_firebase = False
                # Fall through to FAISS initialization
        
        if not self.use_firebase:
            # Use local FAISS
            # Ensure memory directory exists
            os.makedirs(self.memory_cache_dir, exist_ok=True)
            
            # Create a FAISS index name based on agent ID
            self.memory_index = f"memory_{self.agent_id}"
            
            # Check if memory already exists
            memory_path = os.path.join(self.memory_cache_dir, f"{self.memory_index}.faiss")
            if os.path.exists(memory_path) and os.path.exists(f"{memory_path}.json"):
                # Load existing memory
                self.memory_store = FAISS.load_local(
                    folder_path=self.memory_cache_dir,
                    index_name=self.memory_index,
                    embeddings=self.embeddings
                )
                logger.info(f"Loaded local memory for agent {self.agent_id}")
            else:
                # Create empty memory
                self.memory_store = FAISS.from_documents(
                    documents=[Document(page_content="Memory initialization", metadata={"init": True})],
                    embedding=self.embeddings
                )
                # Save to disk
                self.memory_store.save_local(self.memory_cache_dir, index_name=self.memory_index)
                logger.info(f"Initialized local memory for agent {self.agent_id}")

    async def _load_agent_params(self):
        """Load agent parameters from memory or create default ones."""
        try:
            # Try to retrieve the agent parameters
            filter_by = {
                "type": "agent_params"
            }
            
            # Convert to search filter format
            search_filter = {f"metadata.{k}": v for k, v in filter_by.items()}
            search_filter["metadata.agent_id"] = self.agent_id
            
            # Search vector store
            docs = self.memory_store.similarity_search(
                query="agent parameters",
                k=1,
                filter=search_filter
            )
            
            if docs:
                # Found parameters
                doc = docs[0]
                params_json = doc.metadata.get("params", "{}")
                self.memory_params = json.loads(params_json)
                logger.info(f"Loaded memory parameters for agent {self.agent_id}")
            else:
                # No parameters found, save default ones
                await self._save_agent_params()
        except Exception as e:
            logger.error(f"Error loading agent parameters: {str(e)}")
            # Ensure we have default parameters
            self.memory_params = DEFAULT_MEMORY_PARAMS.copy()
            # Try to save them
            await self._save_agent_params()
    
    async def _save_agent_params(self):
        """Save agent parameters to memory."""
        try:
            # Convert parameters to JSON
            params_json = json.dumps(self.memory_params)
            
            # Create metadata
            metadata = {
                "agent_id": self.agent_id,
                "type": "agent_params",
                "params": params_json,
                "timestamp": time.time()
            }
            
            # Create document with description of parameters
            description = f"Memory parameters for agent {self.agent_id}"
            doc = Document(page_content=description, metadata=metadata)
            
            # Add to memory store (or update existing)
            self.memory_store.add_documents([doc])
            
            # If using local FAISS, save to disk
            if not self.use_firebase:
                self.memory_store.save_local(self.memory_cache_dir, index_name=self.memory_index)
            
            logger.info(f"Saved memory parameters for agent {self.agent_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving agent parameters: {str(e)}")
            return False
    
    async def update_memory_params(self, new_params: Dict[str, Any]) -> bool:
        """
        Update memory parameters for this agent.
        
        Args:
            new_params: Dictionary with parameters to update
            
        Returns:
            True if successful
        """
        try:
            # Update parameters
            self.memory_params.update(new_params)
            
            # Save to memory
            success = await self._save_agent_params()
            
            return success
        except Exception as e:
            logger.error(f"Error updating memory parameters: {str(e)}")
            return False
    
    async def get_memory_params(self) -> Dict[str, Any]:
        """
        Get current memory parameters.
        
        Returns:
            Dictionary with memory parameters
        """
        return self.memory_params.copy()
    
    async def add_document(
        self,
        document: Union[str, Document],
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        category: Optional[str] = None
    ) -> str:
        """
        Add a document to agent memory.
        
        Args:
            document: Text content or Document object
            metadata: Optional metadata dictionary
            doc_id: Optional document ID (generated if not provided)
            category: Optional category to tag the document
            
        Returns:
            Document ID
        """
        # Create document ID if not provided
        if not doc_id:
            doc_id = str(uuid4())
        
        # Process document
        if isinstance(document, str):
            # Create metadata if needed
            if metadata is None:
                metadata = {}
            
            # Add category if provided
            if category:
                metadata["category"] = category
                
            # Add standard metadata
            metadata.update({
                "doc_id": doc_id,
                "agent_id": self.agent_id,
                "timestamp": time.time(),
                "type": "document"
            })
            
            # Auto-categorize if enabled and no category provided
            if self.memory_params.get("auto_categorize", True) and not category:
                # This would be a good place to implement auto-categorization
                # For now, we'll just use a default category
                metadata["category"] = "uncategorized"
            
            # Create document
            doc = Document(page_content=document, metadata=metadata)
        else:
            # Use provided Document but add standard metadata
            doc = document
            if doc.metadata is None:
                doc.metadata = {}
                
            # Add category if provided
            if category:
                doc.metadata["category"] = category
                
            # Add standard metadata
            doc.metadata.update({
                "doc_id": doc_id,
                "agent_id": self.agent_id,
                "timestamp": time.time(),
                "type": "document"
            })
        
        # Add to memory store
        self.memory_store.add_documents([doc])
        
        # If using local FAISS, save to disk
        if not self.use_firebase:
            self.memory_store.save_local(self.memory_cache_dir, index_name=self.memory_index)
        
        logger.info(f"Added document to memory: {doc_id}")
        return doc_id
    
    async def add_data_object(
        self,
        content: Dict[str, Any],
        description: str,
        obj_id: Optional[str] = None,
        category: Optional[str] = None
    ) -> str:
        """
        Add a data object to agent memory.
        
        Args:
            content: Dictionary with data
            description: Text description of the data
            obj_id: Optional object ID (generated if not provided)
            category: Optional category to tag the object
            
        Returns:
            Object ID
        """
        # Create object ID if not provided
        if not obj_id:
            obj_id = str(uuid4())
        
        # Serialize content to JSON string
        content_json = json.dumps(content)
        
        # Create metadata
        metadata = {
            "obj_id": obj_id,
            "agent_id": self.agent_id,
            "timestamp": time.time(),
            "type": "data_object",
            "data": content_json
        }
        
        # Add category if provided
        if category:
            metadata["category"] = category
        elif self.memory_params.get("auto_categorize", True):
            # Auto-categorize
            metadata["category"] = "data"
        
        # Create document
        doc = Document(page_content=description, metadata=metadata)
        
        # Add to memory store
        self.memory_store.add_documents([doc])
        
        # If using local FAISS, save to disk
        if not self.use_firebase:
            self.memory_store.save_local(self.memory_cache_dir, index_name=self.memory_index)
        
        logger.info(f"Added data object to memory: {obj_id}")
        return obj_id
    
    async def retrieve_relevant(
        self,
        query: str,
        k: Optional[int] = None,
        category: Optional[str] = None,
        filter_by: Optional[Dict[str, Any]] = None
    ) -> List[Union[Dict[str, Any], str]]:
        """
        Retrieve most relevant memories for a query.
        
        Args:
            query: Query text
            k: Number of results to retrieve (uses memory_params if not specified)
            category: Optional category to filter by
            filter_by: Optional metadata filter dictionary
            
        Returns:
            List of memories (documents or data objects)
        """
        # Use memory params if k not specified
        if k is None:
            k = self.memory_params.get("max_items_per_query", 5)
        
        # Create filter
        search_filter = {}
        
        # Add agent_id filter
        search_filter["metadata.agent_id"] = self.agent_id
        
        # Add category filter if provided
        if category:
            search_filter["metadata.category"] = category
        
        # Add custom filters if provided
        if filter_by:
            for key, value in filter_by.items():
                search_filter[f"metadata.{key}"] = value
        
        # Expand k if using relevance threshold
        search_k = min(k * 2, 20)  # Get more than needed to filter by relevance
        
        # Search vector store
        docs = self.memory_store.similarity_search(
            query=query,
            k=search_k,
            filter=search_filter
        )
        
        # Process results
        results = []
        for doc in docs:
            metadata = doc.metadata
            doc_type = metadata.get("type", "document")
            
            if doc_type == "data_object":
                # Parse data object
                try:
                    data_json = metadata.get("data", "{}")
                    data = json.loads(data_json)
                    results.append({
                        "type": "data_object",
                        "obj_id": metadata.get("obj_id"),
                        "description": doc.page_content,
                        "category": metadata.get("category"),
                        "timestamp": metadata.get("timestamp"),
                        "data": data
                    })
                except Exception as e:
                    logger.error(f"Error parsing data object: {str(e)}")
                    results.append({
                        "type": "error",
                        "message": "Failed to parse data object",
                        "obj_id": metadata.get("obj_id")
                    })
            elif doc_type == "agent_params":
                # Skip agent parameters in results
                continue
            else:
                # Return document
                results.append({
                    "type": "document",
                    "doc_id": metadata.get("doc_id"),
                    "content": doc.page_content,
                    "category": metadata.get("category"),
                    "timestamp": metadata.get("timestamp"),
                    "metadata": {k: v for k, v in metadata.items() if k not in ["doc_id", "agent_id", "timestamp", "type", "category"]}
                })
        
        # Apply post-processing based on strategy
        results = await self._apply_memory_strategy(results, query)
        
        # Limit to k results
        results = results[:k]
        
        logger.info(f"Retrieved {len(results)} memories for query: {query[:30]}...")
        return results
    
    async def _apply_memory_strategy(
        self, 
        results: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Apply memory retrieval strategy based on agent parameters.
        
        Args:
            results: Initial memory retrieval results
            query: Original query
            
        Returns:
            Processed results based on strategy
        """
        strategy = self.memory_params.get("context_strategy", "recency_weighted")
        
        if strategy == "relevance_only":
            # Just return as is (already sorted by relevance)
            return results
            
        elif strategy == "recency_weighted":
            # Re-rank by combining recency and relevance
            # Since results are already sorted by relevance, we'll weight them
            now = time.time()
            for i, item in enumerate(results):
                # Original relevance score (higher index = lower relevance)
                relevance_score = 1.0 - (i / max(1, len(results)))
                
                # Recency score (1.0 = now, 0.0 = oldest possible)
                timestamp = item.get("timestamp", 0)
                age_in_days = (now - timestamp) / 86400  # Convert to days
                max_age = self.memory_params.get("retention_period", 2592000) / 86400
                recency_score = 1.0 - min(1.0, age_in_days / max_age)
                
                # Combined score (higher = better)
                item["_combined_score"] = (0.7 * relevance_score) + (0.3 * recency_score)
            
            # Sort by combined score
            results.sort(key=lambda x: x.get("_combined_score", 0), reverse=True)
            
            # Remove score field
            for item in results:
                if "_combined_score" in item:
                    del item["_combined_score"]
                    
            return results
            
        elif strategy == "priority_first":
            # Sort by priority categories first, then relevance
            priority_categories = self.memory_params.get("priority_categories", [])
            
            if priority_categories:
                # Split into priority and non-priority
                priority_items = []
                other_items = []
                
                for item in results:
                    category = item.get("category")
                    if category in priority_categories:
                        priority_items.append(item)
                    else:
                        other_items.append(item)
                
                # Combine in order (priority first, then others)
                return priority_items + other_items
            else:
                # No priority categories, return as is
                return results
        
        else:
            # Unknown strategy, return as is
            logger.warning(f"Unknown memory strategy: {strategy}, using default")
            return results
    
    async def get_by_id(
        self,
        item_id: str,
        item_type: str = "document"
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a memory item by ID.
        
        Args:
            item_id: Document or object ID
            item_type: 'document' or 'data_object'
            
        Returns:
            Memory item or None if not found
        """
        # Create filter
        if item_type == "document":
            filter_by = {"doc_id": item_id}
        else:
            filter_by = {"obj_id": item_id}
        
        # Add agent_id filter
        filter_by["agent_id"] = self.agent_id
        
        # Add type filter
        filter_by["type"] = item_type
        
        # Convert to search filter format
        search_filter = {f"metadata.{k}": v for k, v in filter_by.items()}
        
        # Search vector store (using arbitrary query, relying on filter)
        docs = self.memory_store.similarity_search(
            query="Retrieve by ID",
            k=1,
            filter=search_filter
        )
        
        if not docs:
            logger.warning(f"No memory found with ID: {item_id}")
            return None
        
        # Process the result
        doc = docs[0]
        metadata = doc.metadata
        doc_type = metadata.get("type", "document")
        
        if doc_type == "data_object":
            # Parse data object
            try:
                data_json = metadata.get("data", "{}")
                data = json.loads(data_json)
                return {
                    "type": "data_object",
                    "obj_id": metadata.get("obj_id"),
                    "description": doc.page_content,
                    "category": metadata.get("category"),
                    "timestamp": metadata.get("timestamp"),
                    "data": data
                }
            except Exception as e:
                logger.error(f"Error parsing data object: {str(e)}")
                return {
                    "type": "error",
                    "message": "Failed to parse data object",
                    "obj_id": metadata.get("obj_id")
                }
        else:
            # Return document
            return {
                "type": "document",
                "doc_id": metadata.get("doc_id"),
                "content": doc.page_content,
                "category": metadata.get("category"),
                "timestamp": metadata.get("timestamp"),
                "metadata": {k: v for k, v in metadata.items() if k not in ["doc_id", "agent_id", "timestamp", "type", "category"]}
            }
    
    async def list_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        List all memories in a category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of memory item summaries
        """
        # Create filter
        search_filter = {
            "metadata.agent_id": self.agent_id,
            "metadata.category": category
        }
        
        # Search vector store (using arbitrary query, relying on filter)
        docs = self.memory_store.similarity_search(
            query="List by category",
            k=100,  # Upper limit, adjust as needed
            filter=search_filter
        )
        
        # Process results
        results = []
        for doc in docs:
            metadata = doc.metadata
            doc_type = metadata.get("type", "document")
            
            if doc_type == "agent_params":
                # Skip agent parameters
                continue
            elif doc_type == "data_object":
                results.append({
                    "type": "data_object",
                    "obj_id": metadata.get("obj_id"),
                    "description": doc.page_content,
                    "category": category,
                    "timestamp": metadata.get("timestamp")
                })
            else:
                results.append({
                    "type": "document",
                    "doc_id": metadata.get("doc_id"),
                    "preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                    "category": category,
                    "timestamp": metadata.get("timestamp")
                })
        
        logger.info(f"Listed {len(results)} memories in category: {category}")
        return results
    
    async def delete_item(self, item_id: str, item_type: str = "document") -> bool:
        """
        Delete a memory item.
        
        Args:
            item_id: Document or object ID
            item_type: 'document' or 'data_object'
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create filter
            if item_type == "document":
                filter_by = {"doc_id": item_id}
            else:
                filter_by = {"obj_id": item_id}
                
            # Add agent_id filter
            filter_by["agent_id"] = self.agent_id
            
            # Add type filter
            filter_by["type"] = item_type
            
            # Convert to search filter format
            search_filter = {f"metadata.{k}": v for k, v in filter_by.items()}
            
            # Search vector store to get the document
            docs = self.memory_store.similarity_search(
                query="Delete item",
                k=1,
                filter=search_filter
            )
            
            if not docs:
                logger.warning(f"No memory found with ID: {item_id}")
                return False
                
            # Delete from memory store
            if self.use_firebase:
                # For Firebase, we need to get the Firebase document ID
                firebase_store = self.memory_store
                # The delete is handled by the FirebaseVectorStore
                firebase_store.delete([item_id])
            else:
                # For FAISS, we need to rebuild the index without this document
                # Get all documents except the one to delete
                all_docs = self.memory_store.similarity_search(
                    query="All documents",
                    k=1000  # Upper limit, adjust as needed
                )
                
                # Filter out the document to delete
                if item_type == "document":
                    filtered_docs = [d for d in all_docs if d.metadata.get("doc_id") != item_id]
                else:
                    filtered_docs = [d for d in all_docs if d.metadata.get("obj_id") != item_id]
                
                # Create new memory store
                if filtered_docs:
                    self.memory_store = FAISS.from_documents(
                        documents=filtered_docs,
                        embedding=self.embeddings
                    )
                else:
                    # If no documents left, initialize with empty document
                    self.memory_store = FAISS.from_documents(
                        documents=[Document(page_content="Memory initialization", metadata={"init": True})],
                        embedding=self.embeddings
                    )
                
                # Save to disk
                self.memory_store.save_local(self.memory_cache_dir, index_name=self.memory_index)
            
            logger.info(f"Deleted memory item: {item_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting memory item: {str(e)}")
            return False
    
    async def wipe_memory(self, keep_params: bool = True) -> bool:
        """
        Clear all memories for this agent.
        
        Args:
            keep_params: Whether to preserve agent parameters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save parameters if keeping them
            params_backup = None
            if keep_params:
                params_backup = self.memory_params.copy()
            
            if self.use_firebase:
                # For Firebase, use the delete method with no IDs to delete all
                firebase_store = self.memory_store
                firebase_store.delete()
            else:
                # For FAISS, initialize a new empty memory
                self.memory_store = FAISS.from_documents(
                    documents=[Document(page_content="Memory initialization", metadata={"init": True})],
                    embedding=self.embeddings
                )
                # Save to disk
                self.memory_store.save_local(self.memory_cache_dir, index_name=self.memory_index)
            
            # Restore parameters if requested
            if keep_params and params_backup:
                self.memory_params = params_backup
                await self._save_agent_params()
            
            logger.info(f"Wiped all memories for agent {self.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error wiping memory: {str(e)}")
            return False
