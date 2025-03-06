# Simple Web Scraper for AI Agents

A lightweight web scraping tool that extracts structured data and passes it to AI agents for further processing.

## Features

- Browser-based scraping using Playwright for JavaScript rendering and anti-detection
- Simple API for requesting scraped content
- AI agent integration to analyze and transform scraped data
- Local storage of scraped content to minimize repeat requests
- Task queue for handling scraping requests

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
# Edit .env with your settings
```

4. Run the API server:
```bash
uvicorn app.main:app --reload
```

### Usage

#### Request content scraping:
```bash
curl -X POST http://localhost:8000/scrape -H "Content-Type: application/json" -d '{"url":"https://example.com", "selector":"main"}'
```

#### Process with AI agent:
```bash
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{"url":"https://example.com", "question":"What is the main topic of this page?"}'
```

## Project Structure

- `app/`: Main application code
  - `scraper/`: Web scraping components
  - `agents/`: AI agent integration
  - `api/`: API routes and definitions
  - `storage/`: Local data storage logic
  - `main.py`: Application entry point

## License

MIT
