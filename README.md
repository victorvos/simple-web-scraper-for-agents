# Simple Web Scraper for AI Agents

A lightweight web scraping tool that extracts structured data and passes it to AI agents for further processing.

## Features

- **Browser-based scraping** using Playwright:
  - Full JavaScript rendering support
  - Advanced anti-detection techniques
  - Stealth mode to avoid blocking
  - Human-like browsing patterns

- **Efficient caching system**:
  - Stores scraped content locally
  - Minimizes repeated requests to the same pages
  - Configurable time-to-live (TTL) for cache entries

- **Content vectorization**:
  - Converts scraped content into vector embeddings
  - Retrieves only the most relevant content for each query
  - Reduces token usage by 70-90% compared to raw text
  - Improves response quality for large documents

- **AI integration**:
  - Uses OpenAI to analyze web content
  - Can answer questions about scraped content
  - Summarization capabilities
  - Smart chunking and context handling

- **API endpoints**:
  - `/api/scrape` - Extract content from a URL
  - `/api/analyze` - Analyze content with AI
  - `/api/summarize` - Generate summaries of web content
  - `/api/scrape-multiple` - Follow links and scrape multiple pages

## Quick Start

### Setup

1. Clone this repository:
```bash
git clone https://github.com/victorvos/simple-web-scraper-for-agents.git
cd simple-web-scraper-for-agents
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key and other settings
```

4. Create necessary directories:
```bash
mkdir -p data/cache data/vector_cache
```

5. Run the API server:
```bash
uvicorn app.main:app --reload
```

### Usage

#### Test scraping directly:
```bash
# Test a simple scrape
python -m scripts.test_scrape https://en.wikipedia.org/wiki/Web_scraping

# Test AI analysis with vectorization (more efficient)
python -m scripts.test_analyze https://en.wikipedia.org/wiki/Web_scraping "What are the ethical concerns with web scraping?"

# Test AI analysis without vectorization (for comparison)
python -m scripts.test_analyze https://en.wikipedia.org/wiki/Web_scraping "What are the ethical concerns with web scraping?" --no-vector
```

#### Use the API:
```bash
# Request content scraping
curl -X POST http://localhost:8000/api/scrape -H "Content-Type: application/json" -d '{"url":"https://example.com"}'

# Ask a question about a website (with vectorization)
curl -X POST http://localhost:8000/api/analyze -H "Content-Type: application/json" -d '{"url":"https://example.com", "question":"What is this website about?", "use_vectorization":true}'

# Generate a summary of a website
curl -X POST http://localhost:8000/api/summarize -H "Content-Type: application/json" -d '{"url":"https://example.com", "max_length":300}'
```

#### API Documentation:
Once the server is running, you can access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

- `app/`: Main application code
  - `scraper/`: Web scraping components
    - `browser.py`: Playwright-based browser automation with anti-detection
    - `scraper.py`: Core scraping functionality
  - `agents/`: AI agent integration
    - `openai_agent.py`: OpenAI integration for analysis
    - `document_processor.py`: Content vectorization and optimization
  - `api/`: API routes and definitions
  - `storage/`: Local data storage logic
    - `cache.py`: Caching system for scraped content
  - `main.py`: Application entry point
- `scripts/`: Utility scripts
  - `test_scrape.py`: Test script for the scraper
  - `test_analyze.py`: Test script for AI analysis
- `data/`: (Created at runtime)
  - `cache/`: Stores scraped HTML and extracted text
  - `vector_cache/`: Stores vectorized document embeddings
- `requirements.txt`: Dependencies
- `.env.example`: Environment variables template

## How Vectorization Works

The vectorization process improves efficiency and reduces costs by:

1. Splitting the document into smaller chunks
2. Converting each chunk into a vector embedding
3. When a question is asked, finding the most semantically relevant chunks
4. Sending only the most relevant content to the LLM, not the entire document

Benefits include:
- 70-90% reduction in token usage compared to raw text
- Improved response quality for large documents
- Faster responses
- Ability to handle much larger documents than would fit in context window

## Storage and Caching System

The application uses two types of caching to improve performance and reduce API costs:

### Content Cache (`data/cache/`)
- Stores raw HTML and extracted text from scraped websites
- Uses file-based storage with MD5 hashing for cache keys
- Configurable TTL (time-to-live) for cache entries
- Prevents unnecessary repeat requests to the same URLs

### Vector Cache (`data/vector_cache/`)
- Stores FAISS vector indexes with document embeddings
- Each webpage gets its own vector index for semantic search
- Significantly speeds up repeat queries to the same website
- Reduces OpenAI API costs for embedding generation

You can configure cache settings in the `.env` file:
```
CACHE_DIR=./data/cache
CACHE_EXPIRY=86400  # 24 hours in seconds
VECTOR_CACHE_DIR=./data/vector_cache
```

## Configuration Options

All configuration can be done through environment variables in the `.env` file:

### OpenAI Settings
- `OPENAI_API_KEY`: Your OpenAI API key

### Scraping Settings
- `SCRAPE_DELAY`: Time in seconds between actions (default: 2.0)
- `USE_STEALTH_MODE`: Enable stealth techniques (default: true)
- `MAX_PAGES`: Maximum pages to scrape in multi-page mode (default: 3)
- `BROWSER_HEADLESS`: Run browser in headless mode (default: true)

### Vectorization Settings
- `USE_VECTORIZATION`: Enable vectorization by default (default: true)
- `CHUNK_SIZE`: Size of document chunks for vectorization (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

### Server Settings
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)

## License

MIT
