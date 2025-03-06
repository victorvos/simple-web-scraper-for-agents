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

class SummarizeRequest(BaseModel):
    """Request model for summarizing content."""
    url: HttpUrl
    max_length: int = Field(500, ge=100, le=2000, description="Maximum summary length in words")
    use_cache: bool = True

class MultiPageRequest(BaseModel):
    """Request model for multi-page scraping."""
    start_url: HttpUrl
    max_pages: int = Field(3, ge=1, le=10, description="Maximum number of pages to scrape")
    same_domain_only: bool = True
    use_cache: bool = True

# Dependency for OpenAI agent
async def get_openai_agent():
    """Get OpenAI agent instance."""
    try:
        return OpenAIAgent()
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
    """
    try:
        logger.info(f"Analyzing URL: {request.url} with question: {request.question}")
        
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
            question=request.question
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
    """
    try:
        logger.info(f"Summarizing URL: {request.url}")
        
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
            max_length=request.max_length
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
