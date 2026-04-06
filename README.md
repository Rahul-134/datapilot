# DataPilot

DataPilot is an AI-powered data analysis and web scraping platform built with FastAPI. It provides an end-to-end workflow for uploading datasets, cleaning data, running natural language queries against your data, and extracting structured information from any website -- all through a modern browser-based interface.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the Application](#running-the-application)
- [Usage](#usage)
  - [Dashboard -- Upload and Overview](#dashboard----upload-and-overview)
  - [Data Cleaning](#data-cleaning)
  - [Analytics -- Natural Language Queries](#analytics----natural-language-queries)
  - [Web Scraper](#web-scraper)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [License](#license)

---

## Features

### Data Upload and Overview
- Upload CSV and Excel files (.csv, .xlsx, .xls) via drag-and-drop or file picker.
- Instantly view dataset statistics: row count, column count, data types, null values, and duplicate rows.
- Data health score calculated from null and duplicate ratios.
- Column profile table with type badges and null percentage per column.
- Sample data preview of the first five rows.

### Data Cleaning
- Remove duplicate rows from the dataset.
- Handle missing values with configurable strategies:
  - Drop rows containing nulls
  - Fill numeric nulls with mean or median
  - Fill text nulls with a placeholder value
- Live preview of cleaning impact before applying.
- Download the cleaned dataset in CSV or Excel format.
- Reset to the original uploaded file at any time.

### Natural Language Data Analysis
- Query your uploaded data using plain English (e.g., "Show me the top 10 rows by price").
- Powered by the Gemini API -- translates natural language into pandas code, executes it, and returns results.
- View the generated Python/pandas code for transparency and reproducibility.
- Copy generated code to clipboard with one click.
- Voice input support via the Web Speech API (Chrome and Edge).
- Query history for quick re-runs.
- Export query results as CSV or Excel.

### AI-Powered Web Scraper
- **URL Mode**: Paste any URL and describe the data you want. The scraper fetches the page, cleans the HTML, and uses Gemini to extract structured data matching your instruction.
  - Automatic pagination detection with 5 strategies (next link, rel=next, numbered pages, URL patterns, CSS selectors).
  - Consistent column schema enforced across paginated pages.
  - Deduplication of extracted rows.
- **Search Mode**: Describe what data you need in natural language. The system will:
  1. Analyze the query to determine a comprehensive schema (8-12 columns) and extraction instructions.
  2. Discover relevant URLs via Google Custom Search API, DuckDuckGo, or Gemini (cascading fallback with URL relevance scoring).
  3. Scrape each discovered site with the unified schema.
  4. Row-level relevance filtering to keep good data and discard noise.
  5. Merge, align, and deduplicate results from multiple sources.
  6. Supplement with AI-generated data if web sources yield too few rows.
- **Structured Data Extraction**: Automatically extracts JSON-LD schemas, OpenGraph meta tags, and HTML tables as a structured preamble to improve extraction quality.
- **Sub-Page Detail Enrichment**: After extracting listing-page data, the scraper follows links into detail pages to fill in missing or shallow values — turning "iPhone 15, $799" into full specs, descriptions, and ratings.
- **Concurrent Fetching**: Detail pages are fetched in parallel using a thread pool (3 workers) for faster enrichment.
- **URL Relevance Filtering**: Search results are scored for relevance before scraping, preventing wasted API calls on irrelevant pages.
- **Multi-Model Cascade**: Automatically falls through four Gemini models (gemini-2.5-flash → gemini-3.1-flash-lite → gemini-3-flash → gemini-2.5-flash-lite) when a model's rate limit is hit, maximizing uptime.
- **Rate Limit Resilience**: Built-in retry with model switching ensures scraping completes even under heavy API quota pressure.
- Download scraped data as CSV or Excel.

---

## Architecture

```
Browser (HTML/JS)
      |
      | HTTP (REST API)
      |
  FastAPI Server
      |
      +-- Routes (upload, clean, query, scraper)
      |       |
      +-- Services
              |-- file_parser.py     File upload and parsing (CSV/Excel)
              |-- data_cleaner.py    Duplicate removal, null handling
              |-- llm_service.py     NL-to-pandas code generation via Gemini
              |-- scraper_service.py Web scraping, pagination, search mode
```

The frontend is server-rendered using Jinja2 templates and communicates with the backend through JSON API endpoints. Static assets (CSS, JavaScript) are served by FastAPI's static file handler.

---

## Tech Stack

| Layer     | Technology                                               |
|-----------|----------------------------------------------------------|
| Backend   | Python 3, FastAPI, Uvicorn                               |
| AI/LLM    | Google Gemini API (gemini-2.5-flash, gemini-3.1-flash-lite, gemini-3-flash, gemini-2.5-flash-lite — auto-cascade) |
| Data      | pandas, openpyxl, BeautifulSoup, lxml                     |
| Frontend  | HTML, JavaScript, Tailwind CSS (CDN)                      |
| Fonts     | Space Grotesk, Space Mono, Syne (Google Fonts)            |
| HTTP      | requests (scraping), httpx (Gemini SDK)                   |
| Templating| Jinja2                                                    |

---

## Getting Started

### Prerequisites

- Python 3.10 or later
- A Google Gemini API key (obtain one from [Google AI Studio](https://aistudio.google.com/))

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/DataPilot.git
   cd DataPilot
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Create a `.env` file in the project root with your Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
```

### Running the Application

Start the development server:

```bash
uvicorn backend.main:app --reload
```

The application will be available at [http://localhost:8000](http://localhost:8000).

---

## Usage

### Dashboard -- Upload and Overview

Navigate to the root URL. Drag and drop a CSV or Excel file onto the upload zone, or click to browse. After upload, the dashboard displays:

- Row and column counts
- Duplicate and null value counts
- File size
- A data health score (0-100%)
- Column profile with types and null percentages
- A sample data preview

### Data Cleaning

Click **Clean Data** on the dashboard to open the cleaning modal. Select which operations to apply:

- Toggle duplicate removal on or off.
- Toggle null handling and choose a strategy (drop, mean, median, or unknown fill).
- Review the estimated impact in the live preview panel.
- Click **Apply & Clean** to execute. The dashboard statistics update immediately.
- Download the cleaned file or reset to the original upload.

### Analytics -- Natural Language Queries

Click **Start Full Analysis** on the dashboard (or navigate to the Analytics page). Type a question about your data in plain English, or use voice input. The system generates pandas code, executes it against your dataset, and displays the results in a table.

Examples of supported queries:
- "Show top 10 rows by selling price"
- "Count unique values in each column"
- "Average value grouped by category"
- "Filter rows where price is greater than 500"

### Web Scraper

Navigate to the Scraper page from the navigation bar.

**URL Mode**: Enter a target URL and describe what data to extract. Set the number of pages to scrape (1-20). Click **Scrape & Extract Data**.

**Search Mode**: Toggle to Search Mode. Describe the data you need in natural language. Configure the number of websites to discover (1-10) and pages per site (1-10). Click **Search & Extract Data**. The system finds relevant websites via search engines, scrapes them with a unified schema, filters out irrelevant rows, and merges the results. If any Gemini model is rate-limited, the system automatically switches to another available model.

---

## API Reference

| Method | Endpoint              | Description                              |
|--------|-----------------------|------------------------------------------|
| POST   | `/api/upload/`        | Upload a CSV or Excel file               |
| POST   | `/api/clean/`         | Clean the uploaded dataset               |
| GET    | `/api/clean/download` | Download the cleaned dataset             |
| POST   | `/api/clean/reset`    | Reset to the original uploaded file      |
| POST   | `/api/query/`         | Run a natural language query             |
| GET    | `/api/query/download` | Download query results (CSV or Excel)    |
| POST   | `/api/scrape/`        | Scrape a specific URL                    |
| POST   | `/api/scrape/search`  | Search mode -- discover and scrape URLs  |
| GET    | `/api/scrape/download`| Download scraped data (CSV or Excel)     |

---

## Project Structure

```
DataPilot/
|-- backend/
|   |-- main.py                    Application entry point and route registration
|   |-- routes/
|   |   |-- upload.py              File upload endpoint
|   |   |-- clean.py               Data cleaning and download endpoints
|   |   |-- query.py               Natural language query endpoint
|   |   |-- scraper.py             Web scraping endpoints (URL and search modes)
|   |-- services/
|       |-- file_parser.py         CSV/Excel parsing and in-memory storage
|       |-- data_cleaner.py        Duplicate removal and null handling logic
|       |-- llm_service.py         Gemini-powered NL-to-pandas translation
|       |-- scraper_service.py     HTML fetching, cleaning, pagination, search pipeline
|-- frontend/
|   |-- templates/
|   |   |-- index.html             Dashboard -- upload and dataset overview
|   |   |-- analyze.html           Analytics -- natural language query interface
|   |   |-- scraper.html           Web scraper interface (URL and search modes)
|   |-- static/
|       |-- css/
|       |   |-- style.css          Global styles
|       |-- js/
|           |-- app.js             Application JavaScript
|           |-- dataTable.js       Data table rendering
|           |-- speech.js          Speech recognition integration
|-- .env                           Environment variables (not committed)
|-- .gitignore                     Git ignore rules
|-- requirements.txt               Python dependencies
|-- README.md                      This file
```

---

## License

This project is provided as-is for personal and educational use.
