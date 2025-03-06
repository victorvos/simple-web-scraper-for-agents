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

- **Firebase vector database integration**:
  - Persistent cloud storage for vector embeddings
  - Scales across multiple instances and deployments
  - Shared vector embeddings database for distributed scraping
  - Automatic backup and redundancy

- **Agent memory system**:
  - Persistent memory for each agent using vector database
  - Store documents and structured data objects
  - Tag and categorize memory items
  - Semantic search across agent memory
  - Memory context augmentation for AI responses
  - Automatic memory from web scraping results

- **AI integration**:
  - Uses OpenAI to analyze web content
  - Can answer questions about scraped content
  - Summarization capabilities
  - Smart chunking and context handling
  - Context-aware responses using agent memory

- **API endpoints**:
  - `/api/scrape` - Extract content from a URL
  - `/api/analyze` - Analyze content with AI
  - `/api/summarize` - Generate summaries of web content
  - `/api/scrape-multiple` - Follow links and scrape multiple pages
  - `/api/memory/*` - Endpoints for managing agent memory

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
mkdir -p data/cache data/vector_cache data/memory_cache
```

5. Firebase Setup (Optional but Recommended):
   - Create a Firebase project at [https://console.firebase.google.com/](https://console.firebase.google.com/)
   - Set up Firestore Database and Storage in your project
   - Download your Firebase Admin SDK service account key JSON file
   - Save the JSON file as `firebase-credentials.json` in your project root
   - Update your `.env` file with Firebase settings:
     ```
     USE_FIREBASE_VECTORDB=true
     FIREBASE_CREDENTIALS_PATH=./firebase-credentials.json
     FIREBASE_STORAGE_BUCKET=your-project-id.appspot.com
     ```

6. Run the API server:
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

# Test agent memory functionality
python -m scripts.test_memory

# Test simple memory augmentation
python -m scripts.test_memory --simple --url https://en.wikipedia.org/wiki/Large_language_model --question "What are the capabilities and limitations of LLMs?"
```

#### Use the API:
```bash
# Request content scraping
curl -X POST http://localhost:8000/api/scrape -H "Content-Type: application/json" -d '{"url":"https://example.com"}'

# Ask a question about a website (with vectorization)
curl -X POST http://localhost:8000/api/analyze -H "Content-Type: application/json" -d '{"url":"https://example.com", "question":"What is this website about?", "use_vectorization":true}'

# Generate a summary of a website
curl -X POST http://localhost:8000/api/summarize -H "Content-Type: application/json" -d '{"url":"https://example.com", "max_length":300}'

# Add to agent memory
curl -X POST http://localhost:8000/api/memory/add -H "Content-Type: application/json" -d '{"content":"Important information to remember", "content_type":"document", "category":"notes"}'

# Retrieve from agent memory
curl -X POST http://localhost:8000/api/memory/retrieve -H "Content-Type: application/json" -d '{"agent_id":"your-agent-id", "query":"information"}'
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
    - `memory.py`: Agent memory system using vector database
  - `api/`: API routes and definitions
  - `storage/`: Local data storage logic
    - `cache.py`: Caching system for scraped content
    - `firebase_vector_db.py`: Firebase vector database integration
  - `main.py`: Application entry point
- `scripts/`: Utility scripts
  - `test_scrape.py`: Test script for the scraper
  - `test_analyze.py`: Test script for AI analysis
  - `test_firebase_vectors.py`: Test script for Firebase vector database
  - `test_memory.py`: Test script for agent memory functionality
- `data/`: (Created at runtime)
  - `cache/`: Stores scraped HTML and extracted text
  - `vector_cache/`: Stores vectorized document embeddings (local mode only)
  - `memory_cache/`: Stores agent memory (local mode only)
- `requirements.txt`: Dependencies
- `.env.example`: Environment variables template
- `firebase-credentials.json`: Firebase credentials file (you must create this)

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

## Agent Memory System

The agent memory system allows AI agents to maintain persistent knowledge across different conversations and sessions. It leverages the vector database for efficient storage and retrieval of information based on semantic similarity.

### Memory Types

The system supports two main types of memory items:

1. **Documents**: Text-based memory items with metadata
   - Scraped web content
   - Notes and information
   - Custom knowledge entries

2. **Data Objects**: Structured JSON data with a text description
   - Statistics and numeric data
   - Structured information
   - Configuration and preferences

### Memory Features

- **Persistent Storage**: Memory persists across application restarts
- **Semantic Search**: Find relevant memories based on semantic meaning
- **Categorization**: Organize memories by categories
- **Metadata Filtering**: Filter and search by metadata values
- **Firebase Integration**: Scale across multiple instances with cloud storage
- **Automatic Capture**: Automatically save relevant web content to memory
- **Memory Context**: Enhance AI responses with relevant memory items

### Using Memory in Your Application

1. **Initialization**:
   ```python
   from app.agents.memory import AgentMemory
   
   # Create a memory instance for a specific agent
   memory = AgentMemory(agent_id="your-agent-id")
   ```

2. **Adding Documents**:
   ```python
   doc_id = await memory.add_document(
       document="Important information to remember",
       metadata={"source": "user input", "importance": "high"},
       category="notes"
   )
   ```

3. **Adding Data Objects**:
   ```python
   obj_id = await memory.add_data_object(
       content={"key1": "value1", "key2": 42, "key3": [1, 2, 3]},
       description="Configuration settings for the application",
       category="config"
   )
   ```

4. **Retrieving Memory**:
   ```python
   # By query (semantic search)
   memories = await memory.retrieve_relevant(
       query="What are the application settings?",
       k=3
   )
   
   # By ID
   memory = await memory.get_by_id(memory_id="doc-123")
   
   # By category
   memories = await memory.list_by_category(category="notes")
   ```

5. **Deleting Memory**:
   ```python
   success = await memory.delete_item(memory_id="doc-123")
   
   # Clear all memory
   success = await memory.wipe_memory()
   ```

## Storage and Caching System

The application uses multiple types of storage to improve performance and reduce API costs:

### Content Cache (`data/cache/`)
- Stores raw HTML and extracted text from scraped websites
- Uses file-based storage with MD5 hashing for cache keys
- Configurable TTL (time-to-live) for cache entries
- Prevents unnecessary repeat requests to the same URLs

### Vector Storage
The application supports two vector storage backends:

#### Local FAISS Vector Store (`data/vector_cache/`)
- Default option if Firebase is not configured
- Stores FAISS vector indexes with document embeddings locally
- Each webpage gets its own vector index for semantic search
- Fast for single-instance deployment

#### Firebase Vector Database (Cloud-based)
- Recommended for production use
- Stores vector embeddings in Firestore
- Enables sharing vector data across multiple instances
- Provides automatic backup and redundancy
- Scales better for production workloads
- Makes vectors persistent across application restarts

To switch between storage backends, configure in your `.env` file:
```
# Use local FAISS
USE_FIREBASE_VECTORDB=false
# Or use Firebase
USE_FIREBASE_VECTORDB=true
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

### Storage Settings
- `CACHE_DIR`: Directory for content cache (default: ./data/cache)
- `CACHE_EXPIRY`: TTL for cache entries in seconds (default: 86400 - 24h)
- `VECTOR_CACHE_DIR`: Directory for vector embeddings (default: ./data/vector_cache)
- `MEMORY_CACHE_DIR`: Directory for agent memory (default: ./data/memory_cache)

### Firebase Settings
- `USE_FIREBASE_VECTORDB`: Use Firebase for vector storage (default: false)
- `FIREBASE_CREDENTIALS_PATH`: Path to Firebase credentials JSON (default: ./firebase-credentials.json)
- `FIREBASE_STORAGE_BUCKET`: Firebase Storage bucket name (e.g., your-project-id.appspot.com)
- `FIREBASE_COLLECTION_NAME`: Firestore collection name for vectors (default: vector_embeddings)
- `FIREBASE_NAMESPACE`: Namespace for grouping vectors (default: web_scraper)

### Server Settings
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)

## Firebase Setup Instructions

To set up Firebase for vector database storage:

1. **Create a Firebase project:**
   - Go to the [Firebase Console](https://console.firebase.google.com/)
   - Click "Add project" and follow the setup wizard

2. **Set up Firestore:**
   - In your project, go to Firestore Database
   - Click "Create database"
   - Choose "Start in production mode" or "Start in test mode" (for development)
   - Select a location for your database

3. **Set up Storage:**
   - In your project, go to Storage
   - Click "Get started"
   - Choose "Start in production mode" or "Start in test mode"
   - Select a location for your storage bucket

4. **Generate a service account key:**
   - In your project, go to Project Settings > Service accounts
   - Select "Firebase Admin SDK"
   - Click "Generate new private key"
   - Save the JSON file as `firebase-credentials.json` in your project root

5. **Configure your .env file:**
   ```
   USE_FIREBASE_VECTORDB=true
   FIREBASE_CREDENTIALS_PATH=./firebase-credentials.json
   FIREBASE_STORAGE_BUCKET=your-project-id.appspot.com
   FIREBASE_COLLECTION_NAME=vector_embeddings
   FIREBASE_NAMESPACE=web_scraper
   ```

## Security Considerations

When using Firebase:
- Never commit your Firebase credentials to version control
- Add `firebase-credentials.json` to your `.gitignore` file
- Consider using environment variables for production deployment
- Set up appropriate Firestore security rules to restrict access

## License

MIT
