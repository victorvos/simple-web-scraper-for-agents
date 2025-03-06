"""API routes for the web scraper application."""
import logging
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, HttpUrl, Field
from typing import Dict, Any, List, Optional, Union

from app.scraper.scraper import WebScraper
from app.agents.openai_agent import OpenAIAgent
from app.storage.cache import clear_expired_cache

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Request and response models
class ScrapeRequest(BaseModel):
    """Request model for scraping a URL."""
    url: HttpUrl
    selector: Optional[str] = None
    use_cache: bool = True
    extract_links: bool = False
    scroll: bool = True

class AnalyzeRequest(BaseModel):
    """Request model for analyzing content."""
    url: HttpUrl
    question: str = Field(..., description="Question to answer about the content")
    use_cache: bool = True
    use_vectorization: bool = True
    use_memory: bool = True

class SummarizeRequest(BaseModel):
    """Request model for summarizing content."""
    url: HttpUrl
    max_length: int = Field(500, ge=100, le=2000, description="Maximum summary length in words")
    use_cache: bool = True
    use_vectorization: bool = True
    use_memory: bool = True

class MultiPageRequest(BaseModel):
    """Request model for multi-page scraping."""
    start_url: HttpUrl
    max_pages: int = Field(3, ge=1, le=10, description="Maximum number of pages to scrape")
    same_domain_only: bool = True
    use_cache: bool = True

class MemoryAddRequest(BaseModel):
    """Request model for adding to agent memory."""
    agent_id: Optional[str] = None
    content: Union[str, Dict[str, Any]]
    content_type: str = Field("document", description="Type of content: 'document' or 'data_object'")
    description: Optional[str] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MemoryRetrieveRequest(BaseModel):
    """Request model for retrieving from agent memory."""
    agent_id: Optional[str] = None
    query: Optional[str] = None
    memory_id: Optional[str] = None
    category: Optional[str] = None
    max_results: int = Field(5, ge=1, le=20, description="Maximum number of results to return")

class MemoryDeleteRequest(BaseModel):
    """Request model for deleting from agent memory."""
    agent_id: str
    memory_id: str
    memory_type: str = Field("document", description="Type of memory: 'document' or 'data_object'")

class MemoryWipeRequest(BaseModel):
    """Request model for wiping agent memory."""
    agent_id: str
    confirm: bool = Field(False, description="Confirmation that memory should be wiped")
    keep_params: bool = Field(True, description="Whether to keep agent memory parameters")

class MemoryParamsRequest(BaseModel):
    """Request model for updating memory parameters."""
    agent_id: str
    params: Dict[str, Any] = Field(..., description="Memory parameters to update")

class MemoryParamsRetrieveRequest(BaseModel):
    """Request model for retrieving memory parameters."""
    agent_id: str

# Dependency for OpenAI agent
async def get_openai_agent(agent_id: Optional[str] = None):
    """Get OpenAI agent instance."""
    try:
        return OpenAIAgent(agent_id=agent_id)
    except ValueError as e:
        logger.error(f"Failed to initialize OpenAI agent: {str(e)}")
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

@router.post("/scrape", summary="Scrape content from a URL")
async def scrape_content(request: ScrapeRequest):
    """
    Scrape content from a URL with optional CSS selector.
    
    - **url**: URL to scrape
    - **selector**: Optional CSS selector to extract specific content
    - **use_cache**: Whether to use cached results if available
    - **extract_links**: Whether to extract links from the page
    - **scroll**: Whether to scroll the page to load lazy content
    """
    try:
        logger.info(f"Scraping URL: {request.url}")
        
        # Scrape the URL
        result = await WebScraper.scrape_url(
            url=str(request.url),
            selector=request.selector,
            use_cache=request.use_cache,
            extract_links=request.extract_links,
            scroll=request.scroll
        )
        
        # Check if successful
        if not result.get("success", False):
            error = result.get("error", "Unknown error")
            raise HTTPException(status_code=400, detail=f"Scraping failed: {error}")
            
        return result
        
    except Exception as e:
        logger.error(f"Error in scrape_content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze", summary="Analyze content from a URL")
async def analyze_content(
    request: AnalyzeRequest,
    openai_agent: OpenAIAgent = Depends(get_openai_agent)
):
    """
    Analyze content from a URL based on a specific question.
    
    - **url**: URL to analyze
    - **question**: Question to answer about the content
    - **use_cache**: Whether to use cached results if available
    - **use_vectorization**: Whether to use vector search for context optimization (more efficient)
    - **use_memory**: Whether to use agent memory for additional context
    """
    try:
        logger.info(f"Analyzing URL: {request.url} with question: {request.question}")
        logger.info(f"Using vectorization: {request.use_vectorization}")
        
        # Scrape the URL first
        content = await WebScraper.scrape_url(
            url=str(request.url),
            use_cache=request.use_cache
        )
        
        # Check if scraping was successful
        if not content.get("success", False):
            error = content.get("error", "Unknown error")
            raise HTTPException(status_code=400, detail=f"Scraping failed: {error}")
            
        # Analyze the content
        analysis = await openai_agent.analyze_content(
            content=content,
            question=request.question,
            use_vectorization=request.use_vectorization,
            use_memory=request.use_memory
        )
        
        # Check if analysis was successful
        if not analysis.get("success", False):
            error = analysis.get("error", "Unknown error")
            raise HTTPException(status_code=400, detail=f"Analysis failed: {error}")
            
        return analysis
        
    except Exception as e:
        logger.error(f"Error in analyze_content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarize", summary="Summarize content from a URL")
async def summarize_content(
    request: SummarizeRequest,
    openai_agent: OpenAIAgent = Depends(get_openai_agent)
):
    """
    Summarize content from a URL.
    
    - **url**: URL to summarize
    - **max_length**: Maximum summary length in words
    - **use_cache**: Whether to use cached results if available
    - **use_vectorization**: Whether to use vector search for context optimization (more efficient)
    - **use_memory**: Whether to use agent memory for additional context
    """
    try:
        logger.info(f"Summarizing URL: {request.url}")
        logger.info(f"Using vectorization: {request.use_vectorization}")
        
        # Scrape the URL first
        content = await WebScraper.scrape_url(
            url=str(request.url),
            use_cache=request.use_cache
        )
        
        # Check if scraping was successful
        if not content.get("success", False):
            error = content.get("error", "Unknown error")
            raise HTTPException(status_code=400, detail=f"Scraping failed: {error}")
            
        # Summarize the content
        summary = await openai_agent.summarize_content(
            content=content,
            max_length=request.max_length,
            use_vectorization=request.use_vectorization,
            use_memory=request.use_memory
        )
        
        # Check if summarization was successful
        if not summary.get("success", False):
            error = summary.get("error", "Unknown error")
            raise HTTPException(status_code=400, detail=f"Summarization failed: {error}")
            
        return summary
        
    except Exception as e:
        logger.error(f"Error in summarize_content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scrape-multiple", summary="Scrape multiple pages starting from a URL")
async def scrape_multiple_pages(request: MultiPageRequest):
    """
    Scrape multiple pages starting from a URL and following links.
    
    - **start_url**: Starting URL to scrape
    - **max_pages**: Maximum number of pages to scrape
    - **same_domain_only**: Whether to only follow links on the same domain
    - **use_cache**: Whether to use cached results if available
    """
    try:
        logger.info(f"Scraping multiple pages starting from: {request.start_url}")
        
        # Scrape multiple pages
        results = await WebScraper.scrape_multiple_pages(
            start_url=str(request.start_url),
            max_pages=request.max_pages,
            same_domain_only=request.same_domain_only
        )
        
        return {
            "success": True,
            "total_pages": len(results),
            "pages": results
        }
        
    except Exception as e:
        logger.error(f"Error in scrape_multiple_pages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/memory/add", summary="Add content to agent memory")
async def add_to_memory(
    request: MemoryAddRequest,
    openai_agent: OpenAIAgent = Depends(lambda: get_openai_agent(request.agent_id))
):
    """
    Add content to agent memory.
    
    - **agent_id**: Optional agent ID (generated if not provided)
    - **content**: Content to add (text string or data object)
    - **content_type**: Type of content ('document' or 'data_object')
    - **description**: Description for data objects
    - **category**: Optional category label
    - **metadata**: Optional metadata dictionary
    """
    try:
        logger.info(f"Adding to memory for agent {openai_agent.agent_id}")
        
        result = await openai_agent.add_to_memory(
            content=request.content,
            content_type=request.content_type,
            description=request.description,
            category=request.category,
            metadata=request.metadata
        )
        
        if not result.get("success", False):
            error = result.get("error", "Unknown error")
            raise HTTPException(status_code=400, detail=f"Failed to add to memory: {error}")
        
        # Return success with agent ID for subsequent calls
        return {
            "success": True,
            "agent_id": openai_agent.agent_id,
            "memory_id": result.get("memory_id"),
            "memory_type": result.get("type")
        }
        
    except Exception as e:
        logger.error(f"Error adding to memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/memory/retrieve", summary="Retrieve content from agent memory")
async def retrieve_from_memory(
    request: MemoryRetrieveRequest,
    openai_agent: OpenAIAgent = Depends(lambda: get_openai_agent(request.agent_id))
):
    """
    Retrieve content from agent memory.
    
    - **agent_id**: Optional agent ID (uses most recent agent if not provided)
    - **query**: Optional text query to search memory
    - **memory_id**: Optional specific memory ID to retrieve
    - **category**: Optional category to filter by
    - **max_results**: Maximum number of results to return for queries
    """
    try:
        logger.info(f"Retrieving from memory for agent {openai_agent.agent_id}")
        
        result = await openai_agent.get_memory(
            query=request.query,
            memory_id=request.memory_id,
            category=request.category,
            k=request.max_results
        )
        
        if not result.get("success", False):
            error = result.get("error", "Unknown error")
            raise HTTPException(status_code=400, detail=f"Failed to retrieve from memory: {error}")
        
        # Add agent ID to result
        result["agent_id"] = openai_agent.agent_id
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving from memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/memory/delete", summary="Delete content from agent memory")
async def delete_from_memory(
    request: MemoryDeleteRequest,
    openai_agent: OpenAIAgent = Depends(lambda: get_openai_agent(request.agent_id))
):
    """
    Delete content from agent memory.
    
    - **agent_id**: Agent ID
    - **memory_id**: Memory item ID to delete
    - **memory_type**: Type of memory ('document' or 'data_object')
    """
    try:
        logger.info(f"Deleting from memory for agent {openai_agent.agent_id}")
        
        result = await openai_agent.delete_memory(
            memory_id=request.memory_id,
            memory_type=request.memory_type
        )
        
        if not result.get("success", False):
            error = result.get("error", "Unknown error")
            raise HTTPException(status_code=400, detail=f"Failed to delete from memory: {error}")
        
        # Return success
        return {
            "success": True,
            "agent_id": openai_agent.agent_id,
            "message": result.get("message")
        }
        
    except Exception as e:
        logger.error(f"Error deleting from memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/memory/wipe", summary="Wipe all agent memory")
async def wipe_memory(
    request: MemoryWipeRequest,
    openai_agent: OpenAIAgent = Depends(lambda: get_openai_agent(request.agent_id))
):
    """
    Wipe all agent memory.
    
    - **agent_id**: Agent ID
    - **confirm**: Confirmation that memory should be wiped (must be true)
    - **keep_params**: Whether to preserve agent memory parameters
    """
    try:
        if not request.confirm:
            raise HTTPException(status_code=400, detail="Confirmation required to wipe memory")
        
        logger.info(f"Wiping memory for agent {openai_agent.agent_id}")
        
        # Access the memory instance directly to use the keep_params option
        result = await openai_agent.memory.wipe_memory(keep_params=request.keep_params)
        
        if not result:
            raise HTTPException(status_code=400, detail="Failed to wipe memory")
        
        # Return success
        return {
            "success": True,
            "agent_id": openai_agent.agent_id,
            "message": f"Memory wiped successfully" + (" (parameters preserved)" if request.keep_params else "")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error wiping memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/memory/params", summary="Get agent memory parameters")
async def get_memory_params(
    request: MemoryParamsRetrieveRequest,
    openai_agent: OpenAIAgent = Depends(lambda: get_openai_agent(request.agent_id))
):
    """
    Get agent memory parameters.
    
    - **agent_id**: Agent ID
    """
    try:
        logger.info(f"Getting memory parameters for agent {openai_agent.agent_id}")
        
        # Get parameters
        params = await openai_agent.memory.get_memory_params()
        
        # Return parameters
        return {
            "success": True,
            "agent_id": openai_agent.agent_id,
            "params": params
        }
        
    except Exception as e:
        logger.error(f"Error getting memory parameters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/memory/params/update", summary="Update agent memory parameters")
async def update_memory_params(
    request: MemoryParamsRequest,
    openai_agent: OpenAIAgent = Depends(lambda: get_openai_agent(request.agent_id))
):
    """
    Update agent memory parameters.
    
    - **agent_id**: Agent ID
    - **params**: Memory parameters to update
    """
    try:
        logger.info(f"Updating memory parameters for agent {openai_agent.agent_id}")
        
        # Update parameters
        success = await openai_agent.memory.update_memory_params(request.params)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update memory parameters")
        
        # Get updated parameters
        params = await openai_agent.memory.get_memory_params()
        
        # Return updated parameters
        return {
            "success": True,
            "agent_id": openai_agent.agent_id,
            "message": "Memory parameters updated successfully",
            "params": params
        }
        
    except Exception as e:
        logger.error(f"Error updating memory parameters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/clear", summary="Clear expired cache entries")
async def clear_cache(background_tasks: BackgroundTasks):
    """
    Clear all expired cache entries.
    This operation runs in the background.
    """
    try:
        # Add task to background
        background_tasks.add_task(clear_expired_cache)
        
        return {
            "success": True,
            "message": "Cache cleanup started in background"
        }
        
    except Exception as e:
        logger.error(f"Error in clear_cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
