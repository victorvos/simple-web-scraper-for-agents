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

class AgentMemory:
    """
    Context memory for agents using vector database storage.
    
    Features:
    - Store documents and data objects with metadata
    - Retrieve relevant context based on queries
    - Maintain persistent memory across sessions
    - Tag memory objects with categories
    - Filter memory by metadata
    """
    
    def __init__(
        self,
        agent_id: str,
        embeddings_model: Optional[Embeddings] = None,
        use_firebase: bool = USE_FIREBASE_VECTORDB,
        firebase_collection: str = FIREBASE_COLLECTION_NAME,
        firebase_namespace: str = FIREBASE_NAMESPACE,
        memory_cache_dir: str = MEMORY_CACHE_DIR
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
        """
        self.agent_id = agent_id
        self.embeddings = embeddings_model or OpenAIEmbeddings()
        self.use_firebase = use_firebase
        self.firebase_collection = firebase_collection
        self.firebase_namespace = firebase_namespace
        self.memory_cache_dir = memory_cache_dir
        
        # Initialize memory store
        self._initialize_memory_store()
        
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
    
    def add_document(
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
    
    def add_data_object(
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
        
        # Create document
        doc = Document(page_content=description, metadata=metadata)
        
        # Add to memory store
        self.memory_store.add_documents([doc])
        
        # If using local FAISS, save to disk
        if not self.use_firebase:
            self.memory_store.save_local(self.memory_cache_dir, index_name=self.memory_index)
        
        logger.info(f"Added data object to memory: {obj_id}")
        return obj_id
    
    def retrieve_relevant(
        self,
        query: str,
        k: int = 5,
        category: Optional[str] = None,
        filter_by: Optional[Dict[str, Any]] = None
    ) -> List[Union[Dict[str, Any], str]]:
        """
        Retrieve most relevant memories for a query.
        
        Args:
            query: Query text
            k: Number of results to retrieve
            category: Optional category to filter by
            filter_by: Optional metadata filter dictionary
            
        Returns:
            List of memories (documents or data objects)
        """
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
        
        # Search vector store
        docs = self.memory_store.similarity_search(
            query=query,
            k=k,
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
        
        logger.info(f"Retrieved {len(results)} memories for query: {query[:30]}...")
        return results
    
    def get_by_id(
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
    
    def list_by_category(self, category: str) -> List[Dict[str, Any]]:
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
            
            if doc_type == "data_object":
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
    
    def delete_item(self, item_id: str, item_type: str = "document") -> bool:
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
    
    def wipe_memory(self) -> bool:
        """
        Clear all memories for this agent.
        
        Returns:
            True if successful, False otherwise
        """
        try:
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
            
            logger.info(f"Wiped all memories for agent {self.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error wiping memory: {str(e)}")
            return False
