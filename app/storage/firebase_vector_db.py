"""Firebase vector database implementation for the web scraper."""
import os
import logging
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import time
import pickle
import base64

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore, storage
from google.cloud.firestore_v1.base_query import FieldFilter

# LangChain imports
from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from langchain.vectorstores.base import VectorStore

# Configure logging
logger = logging.getLogger(__name__)

class FirebaseVectorStore(VectorStore):
    """
    A vector store implementation that uses Firebase for storage.
    
    Features:
    - Stores vector embeddings in Firestore
    - Stores document content and metadata in Firestore
    - Calculates similarity using cosine similarity in Python
    - Supports caching of embeddings in Firebase Storage for faster loading
    """
    
    def __init__(
        self,
        embedding_model: Embeddings,
        collection_name: str = "vector_embeddings",
        namespace: str = "default",
        firebase_app: Optional[firebase_admin.App] = None,
        index_name: Optional[str] = None
    ):
        """
        Initialize the FirebaseVectorStore.
        
        Args:
            embedding_model: The embedding model to use
            collection_name: The Firestore collection to store vectors in
            namespace: The namespace to use for grouping vectors
            firebase_app: Optional firebase app instance (will initialize if None)
            index_name: Optional index name for this vector store
        """
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.namespace = namespace
        self.index_name = index_name or f"index-{int(time.time())}"
        
        # Initialize Firebase if not already initialized
        if firebase_app is None:
            try:
                # Check if Firebase is already initialized
                firebase_admin.get_app()
                self.firebase_app = firebase_admin.get_app()
            except ValueError:
                # Initialize Firebase
                firebase_credentials_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")
                if not firebase_credentials_path:
                    raise ValueError("FIREBASE_CREDENTIALS_PATH environment variable not set")
                
                cred = credentials.Certificate(firebase_credentials_path)
                self.firebase_app = firebase_admin.initialize_app(cred, {
                    'storageBucket': os.environ.get("FIREBASE_STORAGE_BUCKET")
                })
        else:
            self.firebase_app = firebase_app
        
        # Initialize Firestore
        self.db = firestore.client(app=self.firebase_app)
        
        # Initialize Storage if bucket is configured
        storage_bucket = os.environ.get("FIREBASE_STORAGE_BUCKET")
        if storage_bucket:
            self.bucket = storage.bucket(app=self.firebase_app)
        else:
            self.bucket = None
            logger.warning("Firebase Storage bucket not configured. Embeddings caching disabled.")
        
        logger.info(f"Initialized FirebaseVectorStore with collection '{collection_name}' and namespace '{namespace}'")
    
    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        collection_name: str = "vector_embeddings",
        namespace: str = "default",
        firebase_app: Optional[firebase_admin.App] = None,
        index_name: Optional[str] = None,
        **kwargs
    ) -> "FirebaseVectorStore":
        """
        Create a FirebaseVectorStore from documents.
        
        Args:
            documents: List of documents to store
            embedding: Embedding model to use
            collection_name: Firestore collection name
            namespace: Namespace for grouping vectors
            firebase_app: Optional firebase app instance
            index_name: Optional index name
            
        Returns:
            A FirebaseVectorStore instance
        """
        # Initialize the vector store
        vector_store = cls(
            embedding_model=embedding,
            collection_name=collection_name,
            namespace=namespace,
            firebase_app=firebase_app,
            index_name=index_name
        )
        
        # Add documents
        vector_store.add_documents(documents)
        
        return vector_store
    
    def add_documents(self, documents: List[Document], **kwargs):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        # Add texts and metadatas
        logger.info(f"Adding {len(documents)} documents to FirebaseVectorStore")
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Call add_texts
        return self.add_texts(texts, metadatas, **kwargs)
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 50,
        **kwargs
    ) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dicts
            batch_size: Batch size for processing
            
        Returns:
            List of document IDs
        """
        # Handle default for metadatas
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Ensure same length
        assert len(texts) == len(metadatas), "Number of texts and metadatas must match"
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.embedding_model.embed_documents(texts)
        
        # Process in batches
        document_ids = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            
            # Process batch
            batch_ids = self._add_batch(batch_texts, batch_metadatas, batch_embeddings)
            document_ids.extend(batch_ids)
        
        # Cache embeddings in Firebase Storage if configured
        if self.bucket:
            self._cache_embeddings_to_storage(document_ids, embeddings, texts, metadatas)
        
        return document_ids
    
    def _add_batch(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[str]:
        """
        Add a batch of texts to Firestore.
        
        Args:
            texts: List of texts to add
            metadatas: List of metadata dicts
            embeddings: List of embeddings
            
        Returns:
            List of document IDs
        """
        # Create batch
        batch = self.db.batch()
        document_ids = []
        
        # Add each document
        for text, metadata, embedding in zip(texts, metadatas, embeddings):
            # Generate a document ID
            doc_id = self._generate_document_id(text, metadata)
            document_ids.append(doc_id)
            
            # Create document reference
            doc_ref = self.db.collection(self.collection_name).document(doc_id)
            
            # Add document data
            batch.set(doc_ref, {
                'text': text,
                'metadata': metadata,
                'embedding': embedding,  # Firestore supports arrays
                'namespace': self.namespace,
                'index_name': self.index_name,
                'created_at': firestore.SERVER_TIMESTAMP
            })
        
        # Commit the batch
        batch.commit()
        logger.info(f"Added batch of {len(texts)} documents to Firestore")
        
        return document_ids
    
    def _generate_document_id(self, text: str, metadata: Dict[str, Any]) -> str:
        """
        Generate a unique document ID for a text and metadata.
        
        Args:
            text: The document text
            metadata: The document metadata
            
        Returns:
            A unique document ID
        """
        # Create a string to hash
        to_hash = f"{text}:{json.dumps(metadata, sort_keys=True)}:{self.namespace}:{self.index_name}"
        
        # Generate hash
        return hashlib.md5(to_hash.encode()).hexdigest()
    
    def _cache_embeddings_to_storage(
        self,
        document_ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """
        Cache embeddings to Firebase Storage for faster loading.
        
        Args:
            document_ids: List of document IDs
            embeddings: List of embeddings
            texts: List of texts
            metadatas: List of metadata dicts
        """
        try:
            # Convert to numpy array for efficient storage
            np_embeddings = np.array(embeddings, dtype=np.float32)
            
            # Create cache data
            cache_data = {
                'embeddings': np_embeddings,
                'document_ids': document_ids,
                'texts': texts,
                'metadatas': metadatas,
                'namespace': self.namespace,
                'index_name': self.index_name,
                'created_at': time.time()
            }
            
            # Serialize data
            serialized_data = pickle.dumps(cache_data)
            
            # Upload to storage
            cache_path = f"vector_caches/{self.namespace}/{self.index_name}/cache.pkl"
            blob = self.bucket.blob(cache_path)
            blob.upload_from_string(serialized_data)
            
            logger.info(f"Cached embeddings to Firebase Storage: {cache_path}")
        except Exception as e:
            logger.error(f"Error caching embeddings to Firebase Storage: {str(e)}")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: Optional[int] = None,
        **kwargs
    ) -> List[Document]:
        """
        Perform a similarity search against the vector store.
        
        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter
            fetch_k: Optional number of candidates to fetch (defaults to 3*k)
            
        Returns:
            List of documents sorted by relevance
        """
        # Generate embedding for the query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Call similarity search by vector
        return self.similarity_search_by_vector(
            embedding=query_embedding,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs
        )
    
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: Optional[int] = None,
        **kwargs
    ) -> List[Document]:
        """
        Perform a similarity search using a vector.
        
        Args:
            embedding: Query embedding
            k: Number of results to return
            filter: Optional metadata filter
            fetch_k: Optional number of candidates to fetch (defaults to 3*k)
            
        Returns:
            List of documents sorted by relevance
        """
        # Set default fetch_k if not provided
        fetch_k = fetch_k or (3 * k)
        
        # Fetch all relevant documents from Firestore
        query = self.db.collection(self.collection_name)
        
        # Add namespace filter
        query = query.where(filter=FieldFilter("namespace", "==", self.namespace))
        
        # Add index_name filter if specified
        if self.index_name:
            query = query.where(filter=FieldFilter("index_name", "==", self.index_name))
        
        # Add metadata filters if specified
        if filter:
            for key, value in filter.items():
                if key.startswith("metadata."):
                    # Extract the metadata field name
                    field_name = key[len("metadata."):]
                    query = query.where(filter=FieldFilter(f"metadata.{field_name}", "==", value))
        
        # Execute query
        docs = query.stream()
        
        # Convert to list for processing
        firestore_docs = list(docs)
        logger.info(f"Retrieved {len(firestore_docs)} documents from Firestore")
        
        # Calculate similarities and sort
        similarities = []
        for doc in firestore_docs:
            doc_data = doc.to_dict()
            doc_embedding = doc_data.get('embedding', [])
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(embedding, doc_embedding)
            
            # Add to results
            similarities.append((doc_data, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k
        results = similarities[:k]
        
        # Convert to Documents
        documents = []
        for doc_data, _ in results:
            text = doc_data.get('text', '')
            metadata = doc_data.get('metadata', {})
            documents.append(Document(page_content=text, metadata=metadata))
        
        return documents
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Cosine similarity (between -1 and 1)
        """
        # Convert to numpy arrays for efficient computation
        np_v1 = np.array(v1)
        np_v2 = np.array(v2)
        
        # Calculate dot product
        dot_product = np.dot(np_v1, np_v2)
        
        # Calculate magnitudes
        v1_magnitude = np.linalg.norm(np_v1)
        v2_magnitude = np.linalg.norm(np_v2)
        
        # Calculate similarity
        similarity = dot_product / (v1_magnitude * v2_magnitude)
        
        return similarity
    
    def save_local(self, folder_path: str, index_name: Optional[str] = None):
        """
        Save the index to a local directory (compatibility method).
        
        Args:
            folder_path: Folder to save to
            index_name: Optional index name
        """
        # Firebase stores everything in the cloud, so this is a no-op
        logger.info("FirebaseVectorStore doesn't need local saving - all data is in Firebase")
        return
    
    @classmethod
    def load_local(
        cls,
        folder_path: str,
        index_name: str,
        embeddings: Embeddings,
        **kwargs
    ) -> "FirebaseVectorStore":
        """
        Load a vector store from a local directory (compatibility method).
        
        Args:
            folder_path: Folder to load from
            index_name: Index name
            embeddings: Embedding model to use
            
        Returns:
            A FirebaseVectorStore instance
        """
        # Initialize a new instance with the same index_name
        vector_store = cls(
            embedding_model=embeddings,
            namespace=kwargs.get("namespace", "default"),
            collection_name=kwargs.get("collection_name", "vector_embeddings"),
            index_name=index_name
        )
        
        return vector_store
    
    def delete(self, ids: Optional[List[str]] = None, **kwargs):
        """
        Delete documents from the vector store.
        
        Args:
            ids: Optional list of document IDs to delete
        """
        if ids:
            # Delete specific documents
            batch = self.db.batch()
            
            for doc_id in ids:
                doc_ref = self.db.collection(self.collection_name).document(doc_id)
                batch.delete(doc_ref)
            
            batch.commit()
            logger.info(f"Deleted {len(ids)} documents from FirebaseVectorStore")
        else:
            # Delete all documents in the namespace and index
            query = self.db.collection(self.collection_name)
            query = query.where(filter=FieldFilter("namespace", "==", self.namespace))
            
            if self.index_name:
                query = query.where(filter=FieldFilter("index_name", "==", self.index_name))
            
            # Get all documents
            docs = query.stream()
            
            # Delete in batches
            batch = self.db.batch()
            count = 0
            
            for doc in docs:
                batch.delete(doc.reference)
                count += 1
                
                # Commit in batches of 500 (Firestore limit)
                if count % 500 == 0:
                    batch.commit()
                    batch = self.db.batch()
            
            # Commit any remaining
            if count % 500 != 0:
                batch.commit()
            
            logger.info(f"Deleted {count} documents from FirebaseVectorStore")
            
            # Also delete cache if it exists
            if self.bucket:
                try:
                    cache_path = f"vector_caches/{self.namespace}/{self.index_name}/cache.pkl"
                    blob = self.bucket.blob(cache_path)
                    blob.delete()
                    logger.info(f"Deleted cache from Firebase Storage: {cache_path}")
                except Exception as e:
                    logger.warning(f"Error deleting cache from Firebase Storage: {str(e)}")
