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
# Edit .env with your OpenAI API key
```

4. Run the API server:
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
  - `main.py`: Application entry point
- `scripts/`: Utility scripts
  - `test_scrape.py`: Test script for the scraper
  - `test_analyze.py`: Test script for AI analysis
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

## License

MIT
