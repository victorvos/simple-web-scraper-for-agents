"""Cache system for storing and retrieving scraped content."""
import os
import json
import time
import logging
import hashlib
import aiofiles
from typing import Dict, Any, Optional
from urllib.parse import urlparse

# Configure logging
logger = logging.getLogger(__name__)

# Get environment variables
CACHE_DIR = os.environ.get("CACHE_DIR", "./data/cache")
CACHE_EXPIRY = int(os.environ.get("CACHE_EXPIRY", "86400"))  # 24 hours by default

# Make sure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(url: str, selector: Optional[str] = None) -> str:
    """
    Generate a cache key for a URL and optional selector.
    
    Args:
        url: The URL to generate a key for
        selector: Optional CSS selector to include in the key
        
    Returns:
        A string cache key
    """
    # Parse URL to normalize it
    parsed = urlparse(url)
    
    # Create normalized URL (remove trailing slashes, etc.)
    normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if parsed.query:
        normalized_url += f"?{parsed.query}"
    
    # Create string to hash
    to_hash = normalized_url
    if selector:
        to_hash += f":{selector}"
    
    # Create hash
    key = hashlib.md5(to_hash.encode()).hexdigest()
    
    return key

def get_cache_path(cache_key: str) -> str:
    """
    Get filesystem path for a cache key.
    
    Args:
        cache_key: The cache key
        
    Returns:
        Path to the cache file
    """
    # Use first two chars of hash as directory to avoid too many files in one dir
    subdir = cache_key[:2]
    cache_dir = os.path.join(CACHE_DIR, subdir)
    
    # Ensure directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    # Return full path
    return os.path.join(cache_dir, f"{cache_key}.json")

async def save_to_cache(cache_key: str, data: Dict[str, Any]) -> bool:
    """
    Save data to cache.
    
    Args:
        cache_key: The cache key
        data: The data to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Add timestamp if not present
        if "timestamp" not in data:
            data["timestamp"] = time.time()
        
        # Add cache metadata
        cache_data = {
            "data": data,
            "cache_key": cache_key,
            "cached_at": time.time(),
            "expires_at": time.time() + CACHE_EXPIRY
        }
        
        # Get path
        cache_path = get_cache_path(cache_key)
        
        # Write to file
        async with aiofiles.open(cache_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(cache_data, ensure_ascii=False))
            
        logger.debug(f"Saved data to cache: {cache_key}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving to cache: {str(e)}")
        return False

async def load_from_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """
    Load data from cache if it exists and is not expired.
    
    Args:
        cache_key: The cache key
        
    Returns:
        The cached data if found and not expired, None otherwise
    """
    try:
        # Get path
        cache_path = get_cache_path(cache_key)
        
        # Check if file exists
        if not os.path.exists(cache_path):
            return None
        
        # Read from file
        async with aiofiles.open(cache_path, "r", encoding="utf-8") as f:
            content = await f.read()
            cache_data = json.loads(content)
        
        # Check if expired
        if time.time() > cache_data.get("expires_at", 0):
            logger.debug(f"Cache expired: {cache_key}")
            return None
        
        logger.debug(f"Loaded data from cache: {cache_key}")
        return cache_data.get("data")
        
    except Exception as e:
        logger.error(f"Error loading from cache: {str(e)}")
        return None

async def delete_from_cache(cache_key: str) -> bool:
    """
    Delete data from cache.
    
    Args:
        cache_key: The cache key
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get path
        cache_path = get_cache_path(cache_key)
        
        # Check if file exists
        if not os.path.exists(cache_path):
            return False
        
        # Delete file
        os.unlink(cache_path)
        
        logger.debug(f"Deleted data from cache: {cache_key}")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting from cache: {str(e)}")
        return False

async def clear_expired_cache() -> int:
    """
    Clear all expired cache entries.
    
    Returns:
        Number of entries cleared
    """
    cleared = 0
    
    try:
        # Walk through cache directory
        for root, dirs, files in os.walk(CACHE_DIR):
            for file in files:
                if not file.endswith(".json"):
                    continue
                    
                file_path = os.path.join(root, file)
                
                try:
                    # Read file
                    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                        content = await f.read()
                        cache_data = json.loads(content)
                    
                    # Check if expired
                    if time.time() > cache_data.get("expires_at", 0):
                        # Delete file
                        os.unlink(file_path)
                        cleared += 1
                        
                except Exception as e:
                    logger.error(f"Error processing cache file {file_path}: {str(e)}")
                    continue
                    
        logger.info(f"Cleared {cleared} expired cache entries")
        return cleared
        
    except Exception as e:
        logger.error(f"Error clearing expired cache: {str(e)}")
        return cleared
