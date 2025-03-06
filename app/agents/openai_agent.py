"""OpenAI-based AI agent for analyzing web content."""
import os
import logging
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

# Configure logging
logger = logging.getLogger(__name__)

# Get environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


class OpenAIAgent:
    """Agent for analyzing web content using OpenAI."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI agent.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
        """
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
            
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
        
        # Initialize document processor with same API key
        self.document_processor = DocumentProcessor(
            embeddings_model=OpenAIEmbeddings(openai_api_key=self.api_key)
        )
        
    async def analyze_content(
        self,
        content: Dict[str, Any],
        question: str,
        use_vectorization: bool = True,
        max_tokens: int = 8000
    ) -> Dict[str, Any]:
        """
        Analyze web content based on a specific question.
        
        Args:
            content: Scraped content data
            question: Question to answer about the content
            use_vectorization: Whether to use vector search for context optimization
            max_tokens: Maximum tokens to use
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Extract basic info
            title = content.get("title", "")
            url = content.get("url", "")
            
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

Answer based ONLY on the information provided in the web content. If the information needed to answer the question is not available in the provided content, clearly state that. Do not make up or infer information that isn't directly supported by the text.
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
                    question=question
                )
                
                return {
                    "success": True,
                    "question": question,
                    "answer": response.strip(),
                    "source": {
                        "title": title,
                        "url": url
                    },
                    "vectorization": vectorization_stats
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

Answer based ONLY on the information provided in the web content. If the information needed to answer the question is not available in the provided content, clearly state that. Do not make up or infer information that isn't directly supported by the text.
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
                        question=question
                    )
                    
                    all_results.append(response.strip())
                
                # Combine results if multiple chunks
                if len(all_results) > 1:
                    combined_result = await self._combine_chunk_results(all_results, question)
                else:
                    combined_result = all_results[0] if all_results else ""
                    
                return {
                    "success": True,
                    "question": question,
                    "answer": combined_result,
                    "source": {
                        "title": title,
                        "url": url
                    }
                }
                
        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}")
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
        use_vectorization: bool = True
    ) -> Dict[str, Any]:
        """
        Summarize web content.
        
        Args:
            content: Scraped content data
            max_length: Maximum summary length in words
            use_vectorization: Whether to use vector search for context optimization
            
        Returns:
            Dictionary with summary results
        """
        try:
            # Extract basic info
            title = content.get("title", "")
            url = content.get("url", "")
            
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

Provide a clear and concise summary of this content. Focus on the main points, key information, and important details. Your summary should be no longer than {max_words} words.
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
                    max_words=max_length
                )
                
                return {
                    "success": True,
                    "summary": response.strip(),
                    "source": {
                        "title": title,
                        "url": url
                    },
                    "vectorization": vectorization_stats
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

Provide a clear and concise summary of this content chunk. Focus on the main points, key information, and important details.
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
                        max_words=max(100, max_length // len(chunks))
                    )
                    
                    all_summaries.append(response.strip())
                
                # Combine summaries if multiple chunks
                if len(all_summaries) > 1:
                    final_summary = await self._combine_summaries(all_summaries, title, max_length)
                else:
                    final_summary = all_summaries[0] if all_summaries else ""
                    
                return {
                    "success": True,
                    "summary": final_summary,
                    "source": {
                        "title": title,
                        "url": url
                    }
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
