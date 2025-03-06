"""Document processing and vectorization for efficient LLM usage."""
import os
import logging
from typing import List, Dict, Any, Optional
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# Configure logging
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Process and vectorize documents for efficient LLM usage.
    
    This class handles:
    1. Document chunking
    2. Embeddings generation
    3. Vector database storage
    4. Retrieval of relevant context
    """
    
    def __init__(
        self, 
        embeddings_model: Optional[Embeddings] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        cache_dir: str = "./data/vector_cache"
    ):
        """
        Initialize the document processor.
        
        Args:
            embeddings_model: Model to use for embeddings (defaults to OpenAIEmbeddings)
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            cache_dir: Directory to cache vector stores
        """
        # Initialize embeddings model
        self.embeddings = embeddings_model or OpenAIEmbeddings()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Set cache directory
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Keep a cache of loaded vector stores
        self.vector_stores = {}
    
    async def process_web_content(
        self,
        content: Dict[str, Any],
        store_id: Optional[str] = None
    ) -> FAISS:
        """
        Process web content into vectorized chunks.
        
        Args:
            content: Dictionary with scraped web content
            store_id: Optional ID for the vector store (for caching)
            
        Returns:
            FAISS vector store with the processed content
        """
        # Extract text and metadata
        text = content.get("text", "")
        url = content.get("url", "")
        title = content.get("title", "")
        
        if not text:
            raise ValueError("No text content to process")
        
        # Generate a store ID if not provided
        if not store_id:
            import hashlib
            store_id = hashlib.md5(f"{url}-{title}".encode()).hexdigest()
        
        # Check cache first
        store_path = os.path.join(self.cache_dir, f"{store_id}.faiss")
        if os.path.exists(store_path) and os.path.exists(store_path + ".json"):
            try:
                # Load from disk
                vector_store = FAISS.load_local(
                    folder_path=self.cache_dir,
                    index_name=store_id,
                    embeddings=self.embeddings
                )
                logger.info(f"Loaded vector store from cache: {store_id}")
                return vector_store
            except Exception as e:
                logger.warning(f"Failed to load vector store from cache: {e}")
        
        # Create documents from the content
        metadata = {
            "url": url,
            "title": title,
            "source": "web_scraping"
        }
        
        # Split text into chunks
        texts = self.text_splitter.split_text(text)
        logger.info(f"Split content into {len(texts)} chunks")
        
        # Create documents with metadata
        documents = []
        for i, chunk in enumerate(texts):
            doc = Document(
                page_content=chunk,
                metadata={
                    **metadata,
                    "chunk": i,
                    "chunk_id": f"{store_id}-{i}"
                }
            )
            documents.append(doc)
        
        # Create vector store
        vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save to disk
        vector_store.save_local(self.cache_dir, index_name=store_id)
        logger.info(f"Saved vector store to cache: {store_id}")
        
        return vector_store
    
    async def retrieve_relevant_context(
        self,
        query: str,
        vector_store: FAISS,
        num_chunks: int = 5
    ) -> List[str]:
        """
        Retrieve relevant context chunks based on a query.
        
        Args:
            query: Question or query to match context for
            vector_store: FAISS vector store to search in
            num_chunks: Number of context chunks to retrieve
            
        Returns:
            List of relevant text chunks
        """
        # Search for similar documents
        docs = vector_store.similarity_search(query, k=num_chunks)
        
        # Extract text from documents
        context_chunks = [doc.page_content for doc in docs]
        
        return context_chunks
    
    async def get_optimized_context(
        self,
        content: Dict[str, Any],
        query: str,
        store_id: Optional[str] = None,
        num_chunks: int = 5
    ) -> Dict[str, Any]:
        """
        Get optimized context for a query from web content.
        
        Args:
            content: Dictionary with scraped web content
            query: Question or query to match context for
            store_id: Optional ID for the vector store (for caching)
            num_chunks: Number of context chunks to retrieve
            
        Returns:
            Dictionary with optimized context and metadata
        """
        # Process content
        vector_store = await self.process_web_content(content, store_id)
        
        # Retrieve relevant context
        context_chunks = await self.retrieve_relevant_context(
            query=query,
            vector_store=vector_store,
            num_chunks=num_chunks
        )
        
        # Create optimized context
        optimized_context = "\n\n".join(context_chunks)
        
        result = {
            "url": content.get("url", ""),
            "title": content.get("title", ""),
            "optimized_text": optimized_context,
            "num_chunks": len(context_chunks),
            "original_length": len(content.get("text", "")),
            "optimized_length": len(optimized_context)
        }
        
        logger.info(
            f"Optimized context: {result['optimized_length']} chars vs " +
            f"{result['original_length']} chars ({result['num_chunks']} chunks)"
        )
        
        return result
