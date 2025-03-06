"""Document processing and vectorization for efficient LLM usage."""
import os
import logging
from typing import List, Dict, Any, Optional
import json
import hashlib

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# Configure logging
logger = logging.getLogger(__name__)

# Get environment variables
VECTOR_CACHE_DIR = os.environ.get("VECTOR_CACHE_DIR", "./data/vector_cache")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))

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
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        cache_dir: str = VECTOR_CACHE_DIR
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
        
        logger.info(f"Document processor initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        logger.info(f"Vector cache directory: {cache_dir}")
    
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
            store_id = hashlib.md5(f"{url}-{title}".encode()).hexdigest()
        
        # If already in memory cache, return it
        if store_id in self.vector_stores:
            logger.info(f"Using in-memory vector store: {store_id}")
            return self.vector_stores[store_id]
        
        # Check disk cache
        store_path = os.path.join(self.cache_dir, f"{store_id}.faiss")
        if os.path.exists(store_path) and os.path.exists(f"{store_path}.json"):
            try:
                # Load from disk
                vector_store = FAISS.load_local(
                    folder_path=self.cache_dir,
                    index_name=store_id,
                    embeddings=self.embeddings
                )
                logger.info(f"Loaded vector store from cache: {store_id}")
                
                # Cache in memory
                self.vector_stores[store_id] = vector_store
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
        
        logger.info(f"Creating vector embeddings for {len(documents)} chunks...")
        
        # Create vector store
        vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save to disk
        vector_store.save_local(self.cache_dir, index_name=store_id)
        logger.info(f"Saved vector store to cache: {store_id}")
        
        # Cache in memory
        self.vector_stores[store_id] = vector_store
        
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
        logger.info(f"Retrieving {num_chunks} chunks most relevant to: {query[:50]}...")
        
        # Search for similar documents
        docs = vector_store.similarity_search(query, k=num_chunks)
        
        # Extract text from documents
        context_chunks = [doc.page_content for doc in docs]
        
        # Log chunk numbers for debugging
        chunk_ids = [doc.metadata.get("chunk", "unknown") for doc in docs]
        logger.info(f"Retrieved chunks {chunk_ids}")
        
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
        try:
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
            
            # Calculate token estimate (rough approximation: 4 chars ≈ 1 token)
            original_tokens = result["original_length"] // 4
            optimized_tokens = result["optimized_length"] // 4
            tokens_saved = original_tokens - optimized_tokens
            
            logger.info(
                f"Optimized context: {result['optimized_length']} chars vs " +
                f"{result['original_length']} chars ({result['num_chunks']} chunks)"
            )
            logger.info(f"Estimated tokens saved: {tokens_saved} (~{tokens_saved * 0.0002:.2f} USD)")
            
            return result
        except Exception as e:
            logger.error(f"Error in get_optimized_context: {str(e)}")
            # If vectorization fails, return the full text as fallback
            return {
                "url": content.get("url", ""),
                "title": content.get("title", ""),
                "optimized_text": content.get("text", ""),
                "num_chunks": 1,
                "original_length": len(content.get("text", "")),
                "optimized_length": len(content.get("text", "")),
                "error": str(e)
            }
