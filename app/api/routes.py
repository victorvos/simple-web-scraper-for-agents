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

class MemoryParamsResetRequest(BaseModel):
    """Request model for resetting memory parameters to defaults."""
    agent_id: str

@router.post("/memory/params/reset", summary="Reset agent memory parameters to defaults")
async def reset_memory_params(
    request: MemoryParamsResetRequest,
    openai_agent: OpenAIAgent = Depends(lambda: get_openai_agent(request.agent_id))
):
    """
    Reset agent memory parameters to default values.
    
    - **agent_id**: Agent ID
    """
    try:
        logger.info(f"Resetting memory parameters for agent {openai_agent.agent_id}")
        
        # Reset parameters
        result = await openai_agent.reset_memory_params()
        
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail="Failed to reset memory parameters")
        
        # Return updated parameters
        return {
            "success": True,
            "agent_id": openai_agent.agent_id,
            "message": "Memory parameters reset to defaults",
            "params": result.get("params", {})
        }
        
    except Exception as e:
        logger.error(f"Error resetting memory parameters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
