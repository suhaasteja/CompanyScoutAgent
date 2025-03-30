# Able AI Assistant

A chatbot application that uses AI to answer questions about Able based on content scraped from their website. The project includes web scraping capabilities, document processing, vector storage with Chroma, and a Streamlit-based chat interface.

## Components

### 1. Web Crawler (Domain Crawler) - crawler.py

A Scrapy-based crawler that systematically visits and indexes pages within a domain:

- Customizable crawl depth
- Support for staying within a specific domain
- Exports results to CSV (URLs, titles, etc.)

### 2. Content Scraper - scraper.py

Processes URLs from the crawler and extracts structured content:

- Uses docling to convert web pages to markdown
- Cleans and normalizes content
- Saves results to CSV (URL, title, markdown content)

### 3. Vector Database Setup - vectordb_setup.py

Processes scraped content and prepares it for semantic search:

- Splits content into manageable chunks
- Generates embeddings using OpenAI's embedding model
- Stores vectors in a Chroma database for efficient retrieval

### 4. Streamlit Chat Interface - retriever.py

Provides a user-friendly chat interface for interacting with Able's information:

- Maintains conversation history
- Uses semantic search to find relevant content
- Formats prompts with proper context for the AI assistant
- Presents responses in a clean chat interface

## Setup and Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your environment variables:
   - Create a `.env` file with your `OPENAI_API_KEY`
   - Or set the environment variable manually

## Usage

### Running the Web Crawler

```bash
python domain_crawler.py https://able.com --max-depth 2 --output-file domain_urls.csv
```

### Processing and Scraping Content

```bash
python scraper.py
```

### Building the Vector Database

```bash
python vectordb_setup.py
```

### Starting the Chatbot Interface

```bash
streamlit run retriever.py
```

### Preview

![Preview](./able_ai_assistant-preview.png)