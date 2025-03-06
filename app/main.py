"""Main application entry point."""
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Simple Web Scraper for AI Agents",
    description="A lightweight web scraping tool that extracts structured data and passes it to AI agents for further processing",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers here to avoid circular imports
from app.api.routes import router as api_router

# Include routers
app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Simple Web Scraper API is running",
        "docs_url": "/docs",
        "api_prefix": "/api"
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Execute startup tasks."""
    logger.info("Starting Simple Web Scraper service")
    
    # Ensure storage directories exist
    cache_dir = os.environ.get("CACHE_DIR", "./data/cache")
    vector_cache_dir = os.environ.get("VECTOR_CACHE_DIR", "./data/vector_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(vector_cache_dir, exist_ok=True)
    
    # Check Firebase configuration if enabled
    use_firebase = os.environ.get("USE_FIREBASE_VECTORDB", "false").lower() == "true"
    if use_firebase:
        firebase_creds_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")
        if not firebase_creds_path or not os.path.exists(firebase_creds_path):
            logger.warning(
                f"Firebase is enabled but credentials file not found at {firebase_creds_path}. "
                "Vector database will fall back to local FAISS."
            )
            os.environ["USE_FIREBASE_VECTORDB"] = "false"
        else:
            logger.info(f"Using Firebase for vector database storage with credentials at {firebase_creds_path}")
    
    # Log configuration
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Vector cache directory: {vector_cache_dir}")
    logger.info(f"Scrape delay: {os.environ.get('SCRAPE_DELAY', '2.0')}s")
    logger.info(f"Stealth mode: {os.environ.get('USE_STEALTH_MODE', 'true')}")
    logger.info(f"Using vectorization: {os.environ.get('USE_VECTORIZATION', 'true')}")
    logger.info(f"Using Firebase vector database: {os.environ.get('USE_FIREBASE_VECTORDB', 'false')}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Execute shutdown tasks."""
    logger.info("Shutting down Simple Web Scraper service")
    
    # Close any active browser instances or connections
    from app.scraper.browser import close_browser
    await close_browser()
