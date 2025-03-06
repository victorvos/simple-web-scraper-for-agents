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
from app.agents.memory import AgentMemory

# Configure logging
logger = logging.getLogger(__name__)

# Get environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
USE_FIREBASE_VECTORDB = os.environ.get("USE_FIREBASE_VECTORDB", "false").lower() == "true"


class OpenAIAgent:
    """Agent for analyzing web content using OpenAI."""
    
    def __init__(self, api_key: Optional[str] = None, agent_id: Optional[str] = None):
        """
        Initialize the OpenAI agent.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            agent_id: Unique identifier for this agent (auto-generated if not provided)
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
            use_firebase=USE_FIREBASE_VECTORDB
        )
        
        logger.info(f"Initialized OpenAI agent with ID: {self.agent_id}")
    
    async def analyze_content(
        self,
        content: Dict[str, Any],
        question: str,
        use_vectorization: bool = True,
        max_tokens: int = 8000,
        use_memory: bool = True,
        memory_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze web content based on a specific question.
        
        Args:
            content: Scraped content data
            question: Question to answer about the content
            use_vectorization: Whether to use vector search for context optimization
            max_tokens: Maximum tokens to use
            use_memory: Whether to access agent memory for context
            memory_query: Custom query for memory retrieval (defaults to the question)
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Extract basic info
            title = content.get("title", "")
            url = content.get("url", "")
            
            # Get memory context if enabled
            memory_context = ""
            memory_items = []
            
            if use_memory:
                memory_query = memory_query or question
                memory_items = await self._retrieve_memory_context(memory_query)
                if memory_items:
                    # Format memory items for context
                    memory_context = self._format_memory_for_context(memory_items)
                    logger.info(f"Added {len(memory_items)} memory items to context")
            
            if use_vectorization:
                # Use vectorization for more efficient context
                logger.info(f"Using vectorized search for query: {question}")
                
                # Get optimized context
                context_data = await self.document_processor.get_optimized_context(
                    content=content,
                    query=question,
                    num_chunks=3  # Adjust based on question complexity
                )
                
                # Use the optimized text
                text_to_analyze = context_data["optimized_text"]
                
                # Add vectorization stats to result
                vectorization_stats = {
                    "original_length": context_data["original_length"],
                    "optimized_length": context_data["optimized_length"],
                    "compression_ratio": round(
                        context_data["optimized_length"] / max(1, context_data["original_length"]), 2
                    )
                }
                
                logger.info(
                    f"Context optimized: {vectorization_stats['compression_ratio'] * 100}% " +
                    f"of original size ({vectorization_stats['optimized_length']} chars)"
                )
                
                # Generate answer using optimized context
                system_template = """You are a helpful AI assistant that analyzes web content. Your task is to answer questions about the provided web content accurately and objectively.

Web Content Title: {title}
Web Content URL: {url}

The following is the most relevant content extracted from the webpage for answering the question:

{content}

{memory_context}

Answer based primarily on the information provided in the web content, but you may also refer to the agent's memory if relevant. If the information needed to answer the question is not available in the provided content or memory, clearly state that. Do not make up or infer information that isn't directly supported by the text.
"""

                human_template = "{question}"
                
                chat_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(system_template),
                    HumanMessagePromptTemplate.from_template(human_template)
                ])
                
                # Create chain
                chain = LLMChain(llm=self.llm, prompt=chat_prompt)
                
                # Run the chain
                response = await chain.arun(
                    title=title,
                    url=url,
                    content=text_to_analyze,
                    question=question,
                    memory_context=memory_context
                )
                
                # Save this content to memory if it's meaningful
                # Only save web content with vectorization to avoid duplicates
                if len(text_to_analyze) > 100:
                    doc_id = await self._save_content_to_memory(
                        content=text_to_analyze,
                        url=url,
                        title=title,
                        category="web_content"
                    )
                    logger.info(f"Saved content to memory with ID: {doc_id}")
                
                return {
                    "success": True,
                    "question": question,
                    "answer": response.strip(),
                    "source": {
                        "title": title,
                        "url": url
                    },
                    "vectorization": vectorization_stats,
                    "memory_used": len(memory_items) > 0
                }
                
            else:
                # Fall back to the old method for simpler queries or when vectorization is disabled
                text = content.get("text", "")
                
                if not text:
                    return {
                        "success": False,
                        "error": "No text content to analyze"
                    }
                
                # Split long text into chunks
                chunks = self.text_splitter.split_text(text)
                logger.info(f"Split content into {len(chunks)} chunks")
                
                # If multiple chunks, process them separately
                all_results = []
                
                # Process each chunk
                for i, chunk in enumerate(chunks):
                    # Create a prompt for this chunk
                    system_template = """You are a helpful AI assistant that analyzes web content. Your task is to answer questions about the provided web content accurately and objectively.
                    
Web Content Title: {title}
Web Content URL: {url}
Content Chunk {chunk_num} of {total_chunks}:
{content_chunk}

{memory_context}

Answer based primarily on the information provided in the web content, but you may also refer to the agent's memory if relevant. If the information needed to answer the question is not available in the provided content or memory, clearly state that. Do not make up or infer information that isn't directly supported by the text.
"""
                    
                    human_template = "{question}"
                    
                    chat_prompt = ChatPromptTemplate.from_messages([
                        SystemMessagePromptTemplate.from_template(system_template),
                        HumanMessagePromptTemplate.from_template(human_template)
                    ])
                    
                    # Create chain
                    chain = LLMChain(llm=self.llm, prompt=chat_prompt)
                    
                    # Run the chain
                    response = await chain.arun(
                        title=title,
                        url=url,
                        chunk_num=i+1,
                        total_chunks=len(chunks),
                        content_chunk=chunk,
                        question=question,
                        memory_context=memory_context
                    )
                    
                    all_results.append(response.strip())
                
                # Combine results if multiple chunks
                if len(all_results) > 1:
                    combined_result = await self._combine_chunk_results(all_results, question)
                else:
                    combined_result = all_results[0] if all_results else ""
                
                # Save first chunk to memory (to avoid saving too much similar content)
                if chunks and len(chunks[0]) > 100:
                    doc_id = await self._save_content_to_memory(
                        content=chunks[0],
                        url=url,
                        title=title,
                        category="web_content"
                    )
                    logger.info(f"Saved content chunk to memory with ID: {doc_id}")
                    
                return {
                    "success": True,
                    "question": question,
                    "answer": combined_result,
                    "source": {
                        "title": title,
                        "url": url
                    },
                    "memory_used": len(memory_items) > 0
                }
                
        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _save_content_to_memory(
        self,
        content: str,
        url: str,
        title: str,
        category: str = "web_content"
    ) -> str:
        """
        Save content to agent memory.
        
        Args:
            content: Content text
            url: Source URL
            title: Content title
            category: Memory category
            
        Returns:
            Document ID
        """
        # Create metadata
        metadata = {
            "url": url,
            "title": title,
            "source": "web_scraper"
        }
        
        # Generate a unique doc_id based on content and URL
        content_hash = hash(content + url) % 100000
        doc_id = f"content-{content_hash}"
        
        # Save to memory
        return await self.memory.add_document(
            document=content,
            metadata=metadata,
            doc_id=doc_id,
            category=category
        )
    
    async def _retrieve_memory_context(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from memory.
        
        Args:
            query: Query to search memory with
            k: Number of memory items to retrieve
            
        Returns:
            List of memory items
        """
        return await self.memory.retrieve_relevant(query=query, k=k)
    
    def _format_memory_for_context(self, memory_items: List[Dict[str, Any]]) -> str:
        """
        Format memory items for inclusion in prompt context.
        
        Args:
            memory_items: List of memory items from retrieve_relevant
            
        Returns:
            Formatted memory context string
        """
        if not memory_items:
            return ""
        
        memory_sections = []
        memory_sections.append("\n\nAGENT MEMORY:")
        
        for i, item in enumerate(memory_items):
            item_type = item.get("type", "unknown")
            
            if item_type == "document":
                # Format document
                memory_sections.append(f"Memory {i+1} (Document):")
                if "title" in item.get("metadata", {}):
                    memory_sections.append(f"Title: {item['metadata']['title']}")
                if "url" in item.get("metadata", {}):
                    memory_sections.append(f"Source: {item['metadata']['url']}")
                memory_sections.append(f"Content: {item['content']}")
            
            elif item_type == "data_object":
                # Format data object
                memory_sections.append(f"Memory {i+1} (Data):")
                memory_sections.append(f"Description: {item['description']}")
                # Convert data to formatted string
                data_str = json.dumps(item['data'], indent=2)
                memory_sections.append(f"Data: {data_str}")
        
        return "\n".join(memory_sections)
    
    async def add_to_memory(
        self,
        content: Union[str, Dict[str, Any]],
        content_type: str = "document",
        description: Optional[str] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add content to agent memory.
        
        Args:
            content: Content to add (text or data object)
            content_type: Type of content ('document' or 'data_object')
            description: Description for data objects
            category: Optional category
            metadata: Optional metadata
            
        Returns:
            Result including the memory item ID
        """
        try:
            if content_type == "document":
                if not isinstance(content, str):
                    return {
                        "success": False,
                        "error": "Document content must be a string"
                    }
                
                doc_id = await self.memory.add_document(
                    document=content,
                    metadata=metadata or {},
                    category=category
                )
                
                return {
                    "success": True,
                    "memory_id": doc_id,
                    "type": "document"
                }
                
            elif content_type == "data_object":
                if not isinstance(content, dict):
                    return {
                        "success": False,
                        "error": "Data object content must be a dictionary"
                    }
                
                if not description:
                    return {
                        "success": False,
                        "error": "Description is required for data objects"
                    }
                
                obj_id = await self.memory.add_data_object(
                    content=content,
                    description=description,
                    category=category
                )
                
                return {
                    "success": True,
                    "memory_id": obj_id,
                    "type": "data_object"
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Invalid content type: {content_type}. Must be 'document' or 'data_object'."
                }
        
        except Exception as e:
            logger.error(f"Error adding to memory: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_memory(
        self,
        query: Optional[str] = None,
        memory_id: Optional[str] = None,
        category: Optional[str] = None,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve content from agent memory.
        
        Args:
            query: Text query to search memory (optional)
            memory_id: Specific memory ID to retrieve (optional)
            category: Category to filter by (optional)
            k: Number of results to return for queries
            
        Returns:
            Dictionary with memory items
        """
        try:
            # Case 1: Retrieve by ID
            if memory_id:
                item = await self.memory.get_by_id(memory_id)
                if not item:
                    return {
                        "success": False,
                        "error": f"No memory found with ID: {memory_id}"
                    }
                
                return {
                    "success": True,
                    "memory": item
                }
            
            # Case 2: List by category
            elif category and not query:
                items = await self.memory.list_by_category(category)
                return {
                    "success": True,
                    "category": category,
                    "count": len(items),
                    "memories": items
                }
            
            # Case 3: Search by query
            elif query:
                filter_by = {}
                if category:
                    filter_by["category"] = category
                
                items = await self.memory.retrieve_relevant(
                    query=query,
                    k=k,
                    filter_by=filter_by
                )
                
                return {
                    "success": True,
                    "query": query,
                    "count": len(items),
                    "memories": items
                }
            
            # Case 4: No criteria provided
            else:
                return {
                    "success": False,
                    "error": "Please provide either query, memory_id, or category"
                }
                
        except Exception as e:
            logger.error(f"Error retrieving memory: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def delete_memory(self, memory_id: str, memory_type: str = "document") -> Dict[str, Any]:
        """
        Delete a memory item.
        
        Args:
            memory_id: Memory item ID
            memory_type: Type of memory ('document' or 'data_object')
            
        Returns:
            Result of deletion operation
        """
        try:
            success = await self.memory.delete_item(memory_id, memory_type)
            
            if success:
                return {
                    "success": True,
                    "message": f"Successfully deleted {memory_type} with ID: {memory_id}"
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to delete {memory_type} with ID: {memory_id}"
                }
                
        except Exception as e:
            logger.error(f"Error deleting memory: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def wipe_memory(self) -> Dict[str, Any]:
        """
        Wipe all agent memory.
        
        Returns:
            Result of wipe operation
        """
        try:
            success = await self.memory.wipe_memory()
            
            if success:
                return {
                    "success": True,
                    "message": f"Successfully wiped all memory for agent {self.agent_id}"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to wipe agent memory"
                }
                
        except Exception as e:
            logger.error(f"Error wiping memory: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _combine_chunk_results(
        self,
        chunk_results: List[str],
        original_question: str
    ) -> str:
        """
        Combine analysis results from multiple chunks.
        
        Args:
            chunk_results: Results from individual chunks
            original_question: The original question
            
        Returns:
            Combined analysis result
        """
        system_template = """You are a helpful AI assistant tasked with combining multiple analysis fragments into a coherent final answer.

Below are separate analysis fragments from different parts of the same web content, all attempting to answer the same question. Your job is to synthesize these fragments into a single coherent answer that addresses the original question.

Original Question: {question}

Fragments from different parts of the content:
{fragments}

Provide a single comprehensive answer based on all the fragments. Eliminate redundancies, resolve any contradictions (mentioning them if significant), and organize the information logically. Your answer should be coherent and read as a single unified response to the original question.
"""
                
        # Format fragments
        formatted_fragments = "\n\n".join([f"Fragment {i+1}:\n{result}" for i, result in enumerate(chunk_results)])
        
        # Create prompt
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("Please combine these fragments into a coherent answer.")
        ])
        
        # Create chain
        chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        
        # Run the chain
        response = await chain.arun(
            question=original_question,
            fragments=formatted_fragments
        )
        
        return response.strip()
        
    async def summarize_content(
        self,
        content: Dict[str, Any],
        max_length: int = 500,
        use_vectorization: bool = True,
        use_memory: bool = True
    ) -> Dict[str, Any]:
        """
        Summarize web content.
        
        Args:
            content: Scraped content data
            max_length: Maximum summary length in words
            use_vectorization: Whether to use vector search for context optimization
            use_memory: Whether to access agent memory for context
            
        Returns:
            Dictionary with summary results
        """
        try:
            # Extract basic info
            title = content.get("title", "")
            url = content.get("url", "")
            
            # Get memory context if enabled
            memory_context = ""
            memory_items = []
            
            if use_memory:
                # For summaries, get memory related to the title or URL topic
                memory_query = f"information about {title}"
                memory_items = await self._retrieve_memory_context(memory_query)
                if memory_items:
                    # Format memory items for context
                    memory_context = self._format_memory_for_context(memory_items)
                    logger.info(f"Added {len(memory_items)} memory items to summary context")
            
            if use_vectorization:
                # For summarization, we want to process the most important parts of the document
                # We use a special query designed to find the main content
                summary_query = f"main points important information about {title}"
                
                # Get optimized context
                context_data = await self.document_processor.get_optimized_context(
                    content=content,
                    query=summary_query,
                    num_chunks=5  # More chunks for summarization
                )
                
                # Use the optimized text
                text_to_summarize = context_data["optimized_text"]
                
                # Add vectorization stats to result
                vectorization_stats = {
                    "original_length": context_data["original_length"],
                    "optimized_length": context_data["optimized_length"],
                    "compression_ratio": round(
                        context_data["optimized_length"] / max(1, context_data["original_length"]), 2
                    )
                }
                
                # Create a prompt for summarization
                system_template = """You are a helpful AI assistant that summarizes web content. Your task is to provide a concise and informative summary of the provided web content.

Web Content Title: {title}
Web Content URL: {url}

The following is the most important content extracted from the webpage:

{content}

{memory_context}

Provide a clear and concise summary of this content. Focus on the main points, key information, and important details. If there's relevant information in the agent's memory, you may incorporate it if it helps provide a more complete summary. Your summary should be no longer than {max_words} words.
"""
                
                human_template = "Summarize this content."
                
                chat_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(system_template),
                    HumanMessagePromptTemplate.from_template(human_template)
                ])
                
                # Create chain
                chain = LLMChain(llm=self.llm, prompt=chat_prompt)
                
                # Run the chain
                response = await chain.arun(
                    title=title,
                    url=url,
                    content=text_to_summarize,
                    max_words=max_length,
                    memory_context=memory_context
                )
                
                # Save this content to memory if it's meaningful
                if len(text_to_summarize) > 100:
                    doc_id = await self._save_content_to_memory(
                        content=text_to_summarize,
                        url=url,
                        title=title,
                        category="web_content"
                    )
                    logger.info(f"Saved summarized content to memory with ID: {doc_id}")
                
                return {
                    "success": True,
                    "summary": response.strip(),
                    "source": {
                        "title": title,
                        "url": url
                    },
                    "vectorization": vectorization_stats,
                    "memory_used": len(memory_items) > 0
                }
            
            else:
                # Fall back to the old method
                text = content.get("text", "")
                
                if not text:
                    return {
                        "success": False,
                        "error": "No text content to summarize"
                    }
                
                # Split long text into chunks
                chunks = self.text_splitter.split_text(text)
                logger.info(f"Split content into {len(chunks)} chunks for summarization")
                
                # Create summaries for each chunk
                all_summaries = []
                
                # Process each chunk
                for i, chunk in enumerate(chunks):
                    # Create a prompt for this chunk
                    system_template = """You are a helpful AI assistant that summarizes web content. Your task is to provide a concise and informative summary of the provided web content.
                    
Web Content Title: {title}
Web Content URL: {url}
Content Chunk {chunk_num} of {total_chunks}:
{content_chunk}

{memory_context}

Provide a clear and concise summary of this content chunk. Focus on the main points, key information, and important details. If there's relevant information in the agent's memory, you may incorporate it if it helps provide a more complete summary.
"""
                    
                    human_template = "Summarize this content in less than {max_words} words."
                    
                    chat_prompt = ChatPromptTemplate.from_messages([
                        SystemMessagePromptTemplate.from_template(system_template),
                        HumanMessagePromptTemplate.from_template(human_template)
                    ])
                    
                    # Create chain
                    chain = LLMChain(llm=self.llm, prompt=chat_prompt)
                    
                    # Run the chain
                    response = await chain.arun(
                        title=title,
                        url=url,
                        chunk_num=i+1,
                        total_chunks=len(chunks),
                        content_chunk=chunk,
                        max_words=max(100, max_length // len(chunks)),
                        memory_context=memory_context
                    )
                    
                    all_summaries.append(response.strip())
                
                # Combine summaries if multiple chunks
                if len(all_summaries) > 1:
                    final_summary = await self._combine_summaries(all_summaries, title, max_length)
                else:
                    final_summary = all_summaries[0] if all_summaries else ""
                
                # Save first chunk to memory
                if chunks and len(chunks[0]) > 100:
                    doc_id = await self._save_content_to_memory(
                        content=chunks[0],
                        url=url,
                        title=title,
                        category="web_content"
                    )
                    logger.info(f"Saved content chunk to memory with ID: {doc_id}")
                    
                return {
                    "success": True,
                    "summary": final_summary,
                    "source": {
                        "title": title,
                        "url": url
                    },
                    "memory_used": len(memory_items) > 0
                }
                
        except Exception as e:
            logger.error(f"Error in AI summarization: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
            
    async def _combine_summaries(
        self,
        summaries: List[str],
        title: str,
        max_length: int = 500
    ) -> str:
        """
        Combine summaries from multiple chunks.
        
        Args:
            summaries: Summaries from individual chunks
            title: Content title
            max_length: Maximum combined summary length in words
            
        Returns:
            Combined summary
        """
        system_template = """You are a helpful AI assistant tasked with combining multiple content summaries into a single coherent summary.

Below are separate summaries from different parts of the same web content with the title: "{title}".

Your job is to synthesize these summaries into a single coherent summary that covers the most important information from the entire content.

Summaries from different parts of the content:
{summaries}

Provide a single comprehensive summary based on all the fragments. Eliminate redundancies, resolve any contradictions, and organize the information logically. Your summary should be coherent, well-structured, and no longer than {max_words} words.
"""
                
        # Format summaries
        formatted_summaries = "\n\n".join([f"Summary {i+1}:\n{summary}" for i, summary in enumerate(summaries)])
        
        # Create prompt
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("Please combine these summaries into a single coherent summary.")
        ])
        
        # Create chain
        chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        
        # Run the chain
        response = await chain.arun(
            title=title,
            summaries=formatted_summaries,
            max_words=max_length
        )
        
        return response.strip()
