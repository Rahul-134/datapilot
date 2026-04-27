# DataPilot

DataPilot is an AI-powered data analysis, web scraping, and machine learning platform built with FastAPI and Streamlit. It provides an end-to-end workflow for uploading datasets, cleaning data, running natural language queries against your data, extracting structured information from any website, and training ML models — all through a modern, light-themed browser interface with a clean editorial design.

---

## Table of Contents

- [Features](#features)
- [ML Model Trainer](#ml-model-trainer)
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
  - [ML Model Trainer Usage](#ml-model-trainer-usage)
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

## ML Model Trainer

The **ML Model Trainer** is a standalone Streamlit application (`ml_trainer/app.py`) accessible directly from the DataPilot navigation bar. It provides a complete, no-code machine learning workflow — from data upload to trained model predictions — without leaving the browser.

### File Upload & Preview
- Upload CSV or Excel files (.csv, .xlsx) via drag-and-drop.
- Instant dataframe preview of the uploaded dataset.
- Session-aware file tracking with confirmation dialog on file removal to prevent accidental data loss.

### Data Cleaning
- **Null value handling** — four strategies: drop rows, fill with mean, fill with median, or fill with a custom value.
- **Duplicate row removal** — detect and remove duplicates with one click.
- **Outlier removal (IQR method)** — select numeric columns, view outlier counts per column, and remove outliers using the interquartile range (Q1 − 1.5·IQR, Q3 + 1.5·IQR).
- Cleaned data preview with live row × column shape indicator.

### Model Training
Supports both **Regression** and **Classification** tasks with the following algorithms:

| Task            | Available Models                                              |
|-----------------|---------------------------------------------------------------|
| Regression      | Linear Regression, Multiple Linear Regression, Polynomial Regression |
| Classification  | Decision Tree, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Random Forest |

**Workflow:**
1. Select task type (Regression or Classification).
2. Choose a model algorithm.
3. Pick independent (X) and dependent (Y/target) columns — supports both numeric and categorical features via automatic one-hot encoding (`pd.get_dummies`).
4. Configure train-test split (10%–40%, default 20%, `random_state=42`).
5. Click **Train Model** — the app automatically searches for optimal hyperparameters.

### Auto Hyperparameter Tuning
Each model performs an automated parameter search to find the best configuration:

| Model               | Parameter Searched       | Range            |
|----------------------|--------------------------|------------------|
| Polynomial Regression| Degree                   | 1–5              |
| Decision Tree        | `max_depth`              | 1–20             |
| KNN                  | `n_neighbors`            | 1–20             |
| SVM                  | `C` (regularization)     | 0.01, 0.1, 1, 10, 100 |
| Random Forest        | `n_estimators`           | 10, 50, 100, 150, 200, 300 |

The best parameter is selected by highest R² (regression) or highest accuracy (classification). Search results are displayed in a table for full transparency.

### Evaluation & Reports
- **Regression metrics**: R² Score, Mean Squared Error (MSE), equation display for simple linear models.
- **Classification metrics**: Accuracy, F1 Score (weighted), full Classification Report (precision, recall, F1 per class), and interactive Confusion Matrix heatmap.
- **Per-model reports**: Click "Get Report" on any trained model to view its full evaluation.

### Prediction
- Select any previously trained model from a dropdown.
- Input feature values via auto-generated form fields (number inputs with min/max/mean for numeric columns, dropdowns for categorical columns).
- Handles preprocessing automatically: one-hot encoding alignment, polynomial feature transformation, SVM scaling (`StandardScaler`).
- Instant prediction result display.

### Code Export
- Generate ready-to-run Python code that reproduces the entire training session for any trained model.
- Includes all imports, data loading, preprocessing, model initialization with best hyperparameters, training, and evaluation.
- View the code inline or download as a `.py` file.

### Visualizations
Seven interactive chart types powered by Plotly:

| Chart Type    | Configurable Options                            |
|---------------|--------------------------------------------------|
| Scatter Plot  | X axis, Y axis, optional color grouping          |
| Histogram     | Column selection, bin count (5–100)              |
| Pie Chart     | Category column, value column                    |
| Bar Chart     | X/Y axes with aggregation (sum, mean, count)    |
| Line Chart    | X/Y axes, optional color grouping               |
| Heatmap       | Multi-column correlation matrix                  |
| Box Plot      | Value column, optional group-by                  |

- 10 color palettes available (Default Blue, Viridis, Plasma, Sunset, Teal, Red-Orange, Green, Rainbow, Pastel, Bold).
- All plotted graphs are stored in the session and displayed in an expandable gallery.
- Graphs can be cleared in bulk.

### Model Comparison
- When two or more models are trained, a comparison table is automatically generated showing all models side-by-side with their metrics.
- Interactive grouped bar charts compare classifier accuracy/F1 or regressor R²/MSE.

---

## Architecture

```
Browser (HTML/JS)                          Streamlit (ML Trainer)
       |                                          |
       | HTTP (REST API)                          | localhost:8501
       |                                          |
   FastAPI Server                          ml_trainer/app.py
       |                                    (standalone app)
       +-- Routes (upload, clean, query, scraper)
       |       |
       +-- Services
               |-- file_parser.py     File upload and parsing (CSV/Excel)
               |-- data_cleaner.py    Duplicate removal, null handling
               |-- llm_service.py     NL-to-pandas code generation via Gemini
               |-- scraper_service.py Web scraping, pagination, search mode
```

The main frontend is server-rendered using Jinja2 templates and communicates with the backend through JSON API endpoints. Static assets (CSS, JavaScript) are served by FastAPI's static file handler. The ML Model Trainer runs as a separate Streamlit application on port 8501, linked from the main navigation bar via a "Train a Model" button.

### Design System

The UI follows a warm, editorial design language:

| Token             | Value                                      |
|-------------------|---------------------------------------------|
| Page Background   | `#F5F4EF` (warm off-white)                 |
| Card Background   | `#FFFFFF` with subtle shadow               |
| Primary Accent    | `#2563EB` (blue-600)                       |
| Secondary Accent  | `#059669` (emerald-600)                    |
| Text – Main       | `#111827` / `#2D3432`                      |
| Text – Muted      | `#6B7280` / `#5A605E`                      |
| Border            | `#E5E7EB`                                  |
| Navigation        | Fixed top glassmorphism (white/85 + backdrop blur) |
| Footer            | Unified light theme (`#ecefec`) with consistent branding |
| Cards             | Rounded corners (12-16 px), light borders  |
| Animations        | Fade-up on load, hover lift, float         |

The ML Trainer uses the same design tokens via custom Streamlit CSS, ensuring visual consistency across both services.

---

## Tech Stack

| Layer      | Technology                                               |
|------------|----------------------------------------------------------|
| Backend    | Python 3, FastAPI, Uvicorn                               |
| AI/LLM     | Google Gemini API (gemini-2.5-flash, gemini-3.1-flash-lite, gemini-3-flash, gemini-2.5-flash-lite — auto-cascade) |
| ML         | scikit-learn (LinearRegression, DecisionTree, KNN, SVM, RandomForest, PolynomialFeatures) |
| Data       | pandas, openpyxl, BeautifulSoup, lxml                    |
| Viz        | Plotly, Matplotlib, Seaborn                              |
| Frontend   | HTML, JavaScript, Tailwind CSS (CDN), Material Symbols   |
| ML UI      | Streamlit                                                |
| Fonts      | Plus Jakarta Sans, Inter, Manrope (Google Fonts)         |
| HTTP       | requests (scraping), httpx (Gemini SDK)                  |
| Templating | Jinja2                                                   |

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

DataPilot consists of two services that run simultaneously:

**1. Start the FastAPI server** (main app — Dashboard, Analytics, Scraper):

```bash
uvicorn backend.main:app --reload
```

This serves the main application at [http://localhost:8000](http://localhost:8000).

**2. Start the Streamlit ML Trainer** (in a separate terminal):

```bash
streamlit run ml_trainer/app.py
```

This launches the ML Model Trainer at [http://localhost:8501](http://localhost:8501).

> **Tip:** The "Train a Model" button in the main DataPilot navbar links directly to the Streamlit app. Both services must be running for the full experience.

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

### Web Scraper (BETA)

Navigate to the Scraper page from the navigation bar. Note: This feature is currently under active development.

**URL Mode**: Enter a target URL and describe what data to extract. Set the number of pages to scrape (1-20). Click **Scrape & Extract Data**.

**Search Mode**: Toggle to Search Mode. Describe the data you need in natural language. Configure the number of websites to discover (1-10) and pages per site (1-10). Click **Search & Extract Data**. The system finds relevant websites via search engines, scrapes them with a unified schema, filters out irrelevant rows, and merges the results. If any Gemini model is rate-limited, the system automatically switches to another available model.

### ML Model Trainer Usage

Click the **Train a Model** button in the navigation bar (available on all pages) to open the ML Trainer in a new tab. The workflow is:

1. **Upload** a CSV or Excel file.
2. **Clean** the data — handle nulls, remove duplicates, remove outliers using the IQR method.
3. **Configure** the model — select task type, algorithm, feature/target columns, and train-test split.
4. **Train** — the app auto-tunes hyperparameters and displays results with metrics.
5. **Predict** — input new values and get instant predictions from any trained model.
6. **Visualize** — create up to 7 chart types with customizable palettes.
7. **Compare** — view a side-by-side comparison table and charts when multiple models are trained.
8. **Export** — generate and download reproducible Python training code.

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

> **Note:** The ML Model Trainer does not expose REST endpoints — it is a standalone Streamlit app with its own UI at `localhost:8501`.

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
|-- ml_trainer/
|   |-- app.py                     Streamlit ML training app (data cleaning, model
|                                  training, prediction, visualization, code export)
|-- .env                           Environment variables (not committed)
|-- .gitignore                     Git ignore rules
|-- requirements.txt               Python dependencies
|-- README.md                      This file
```

---

## License

This project is provided as-is for personal and educational use.
