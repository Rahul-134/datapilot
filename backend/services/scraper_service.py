import requests
import math
import pandas as pd
from bs4 import BeautifulSoup
from google import genai
from dotenv import load_dotenv
from urllib.parse import urljoin
import os
import json
import re

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

DEFAULT_MAX_PAGES = 5  # default pagination pages to follow

def sanitize(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value

def sanitize_row(row: dict) -> dict:
    return {k: sanitize(v) for k, v in row.items()}

def fetch_page(url: str) -> dict:
    """Fetch raw HTML from a URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        return {"success": True, "html": res.text, "status": res.status_code}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. The website took too long to respond."}
    except requests.exceptions.HTTPError as e:
        return {"success": False, "error": f"HTTP error: {e.response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Could not connect to the URL. Check if it is valid and accessible."}
    except Exception as e:
        return {"success": False, "error": str(e)}

def clean_html(html: str) -> str:
    """Strip scripts, styles and return clean readable text.
    Replaces truncated link/image text with their title attributes
    so that full names are preserved for the LLM."""
    soup = BeautifulSoup(html, "lxml")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "iframe", "noscript"]):
        tag.decompose()

    # Fix truncated text: replace inner text of <a> and <img> tags
    # with their title attribute (which usually has the full text)
    for a_tag in soup.find_all("a", title=True):
        title = a_tag["title"].strip()
        if title:
            a_tag.string = title

    for img_tag in soup.find_all("img", alt=True):
        alt = img_tag["alt"].strip()
        if alt:
            img_tag.insert_after(alt)

    return soup.get_text(separator="\n", strip=True)[:15000]  # cap at 15k chars

def detect_next_page(html: str, current_url: str) -> str | None:
    """Look for a 'next' pagination link and return its absolute URL."""
    soup = BeautifulSoup(html, "lxml")

    # Strategy 1: <li class="next"><a href="...">
    next_li = soup.select_one("li.next > a[href]")
    if next_li:
        return urljoin(current_url, next_li["href"])

    # Strategy 2: <a> with text containing "next" (case-insensitive)
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True).lower()
        if text in ("next", "next →", "next »", "next page", "›", "»"):
            return urljoin(current_url, a["href"])

    # Strategy 3: <a rel="next">
    rel_next = soup.find("a", rel="next", href=True)
    if rel_next:
        return urljoin(current_url, rel_next["href"])

    return None

def build_scraper_prompt(user_instruction: str, page_text: str, url: str,
                         expected_columns: list[str] | None = None) -> str:
    column_rule = ""
    if expected_columns:
        cols_str = ", ".join(f'"{c}"' for c in expected_columns)
        column_rule = (
            f"\nIMPORTANT: You MUST use exactly these column names as keys: [{cols_str}].\n"
            f"Do NOT rename, add, or remove any columns. Use these exact keys for every object.\n"
        )

    return f"""You are a data extraction expert. A user wants to extract structured data from a webpage.

URL: {url}
User Instruction: "{user_instruction}"
{column_rule}
Page Content (cleaned):
{page_text}

Your job:
1. Extract the data the user requested from the page content above.
2. Return the data as a JSON array of objects where each object is one row.
3. Each object must have the same keys (column names).
4. Column names should be clean snake_case strings.
5. Return ONLY the JSON array — no explanation, no markdown, no backticks.
6. If no relevant data is found, return an empty array: []

Example output:
[{{"name": "iPhone 15", "price": "$799", "rating": "4.5"}}, {{"name": "Samsung S24", "price": "$699", "rating": "4.3"}}]
"""

def extract_page_data(url: str, html: str, user_instruction: str,
                      expected_columns: list[str] | None = None) -> list | dict:
    """Clean one page's HTML and ask Gemini to extract structured data."""
    page_text = clean_html(html)

    if not page_text.strip():
        return []

    prompt = build_scraper_prompt(user_instruction, page_text, url, expected_columns)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        raw = response.text.strip()
    except Exception as e:
        return {"error": f"Gemini API error: {str(e)}"}

    # Strip markdown fences if model adds them
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    return data

def scrape_url(url: str, user_instruction: str, max_pages: int = DEFAULT_MAX_PAGES) -> dict:
    all_rows = []
    current_url = url
    pages_fetched = 0
    ran_out_of_pages = False

    while current_url and pages_fetched < max_pages:
        # Fetch page
        fetch_result = fetch_page(current_url)
        if not fetch_result["success"]:
            if pages_fetched == 0:
                return {"error": fetch_result["error"]}
            break  # stop pagination on error, keep what we have

        html = fetch_result["html"]
        pages_fetched += 1

        # Extract data from this page (enforce columns from page 1 onward)
        expected_cols = list(all_rows[0].keys()) if all_rows else None
        page_data = extract_page_data(current_url, html, user_instruction, expected_cols)

        if isinstance(page_data, dict) and "error" in page_data:
            if pages_fetched == 1:
                return page_data
            break

        all_rows.extend(page_data)

        # Check for next page
        next_url = detect_next_page(html, current_url)
        if next_url is None and pages_fetched < max_pages:
            ran_out_of_pages = True
        current_url = next_url

    if len(all_rows) == 0:
        return {"error": "No relevant data found for your instruction. Try a different URL or instruction."}

    # Deduplicate rows (by converting dicts to frozensets for comparison)
    seen = set()
    unique_rows = []
    for row in all_rows:
        key = frozenset(sorted((k, str(v)) for k, v in row.items()))
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)

    # Convert to DataFrame summary
    df      = pd.DataFrame(unique_rows)
    columns = df.columns.tolist()
    rows    = [sanitize_row(row) for row in df.to_dict(orient="records")]

    result = {
        "success":      True,
        "url":          url,
        "instruction":  user_instruction,
        "columns":      columns,
        "rows":         rows,
        "total_rows":   len(df),
        "shape":        list(df.shape),
        "pages_scraped": pages_fetched
    }

    if ran_out_of_pages:
        result["warning"] = (
            f"You requested {max_pages} pages, but only {pages_fetched} "
            f"{'page was' if pages_fetched == 1 else 'pages were'} available on this website. "
            f"Showing all available data."
        )

    return result


# ─── SEARCH MODE ──────────────────────────────────────────────────────

def discover_urls(query: str, max_results: int = 3) -> list[str]:
    """Ask Gemini to suggest real, scrapable URLs for a natural-language query."""
    prompt = f"""You are a web research assistant. The user wants to scrape structured data from the internet using a simple HTTP GET request (no JavaScript rendering, no login).

User query: "{query}"

Your job:
1. Suggest up to {max_results} real, publicly accessible website URLs that contain the data the user described.
2. IMPORTANT — the URLs must be scrapable by a simple HTTP client. Prefer:
   - Simple HTML pages with data in tables or lists (e.g. Wikipedia, simple.wikipedia.org)
   - Open data portals (e.g. worldometers.info, worldpopulationreview.com)
   - Public listing sites (e.g. books.toscrape.com, toscrape.com)
   - GitHub pages, wiki pages, or any static HTML site
3. AVOID these types of sites — they block bots or require JavaScript:
   - statista.com, imf.org, worldbank.org/data, bloomberg.com
   - Any site behind a paywall or CAPTCHA
   - Single-page apps (SPAs) that load data via JavaScript/APIs
   - Sites known to return 403/429 for automated requests
4. Each URL should point to a specific page (not just a homepage) where the data can be found in plain HTML.
5. Return ONLY a JSON array of URL strings — no explanation, no markdown, no backticks.

Example output:
["https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)", "https://www.worldometers.info/gdp/gdp-by-country/"]
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        raw = response.text.strip()
    except Exception as e:
        return {"error": f"Gemini API error while discovering URLs: {str(e)}"}

    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        urls = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Failed to parse URL suggestions from the AI model."}

    if not isinstance(urls, list):
        return {"error": "AI model returned unexpected format for URL suggestions."}

    # Filter to only valid-looking URLs
    urls = [u for u in urls if isinstance(u, str) and u.startswith(("http://", "https://"))]

    if not urls:
        return {"error": "The AI model could not find relevant websites for your query. Try rephrasing."}

    return urls[:max_results]


def analyze_query(query: str) -> dict:
    """Analyze the user's query to determine target schema and refined extraction instruction."""
    prompt = f"""You are a data analysis expert. A user wants to search the web and scrape structured data.

User query: "{query}"

Your job:
1. Understand what the user ACTUALLY wants — what specific data fields/columns would be most useful.
2. Create a clear, precise extraction instruction that any page can follow.
3. Define the exact column names (snake_case) that the extracted data should have.

Return a JSON object with exactly these keys:
- "columns": array of snake_case column names that define the target schema (3-8 columns ideal)
- "extraction_instruction": a clear, specific instruction for extracting data from any webpage that contains this type of information. Be very specific about what each column should contain.

Rules:
- Do NOT include generic metadata columns like "source_url" — those are added automatically.
- Focus on the actual data the user wants.
- Column names should be descriptive and consistent.
- The extraction instruction should be specific enough that data from different websites will have the same structure.

Return ONLY the JSON object — no explanation, no markdown, no backticks.

Example for query "Top programming languages with popularity":
{{"columns": ["rank", "language", "popularity_percentage", "paradigm", "year_created"], "extraction_instruction": "Extract a list of programming languages. For each language, extract: its rank/position, the language name, its popularity percentage or index score, its primary programming paradigm, and the year it was created. Only include actual programming languages, not frameworks or tools."}}
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        raw = response.text.strip()
    except Exception:
        return None

    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if isinstance(result, dict) and "columns" in result and "extraction_instruction" in result:
        return result
    return None


def ai_knowledge_fallback(query: str, schema: dict | None = None) -> dict:
    """When all scraping attempts fail, use Gemini's own knowledge to generate the data."""
    column_hint = ""
    if schema and schema.get("columns"):
        cols = ", ".join(f'"{c}"' for c in schema["columns"])
        column_hint = f'\nYou MUST use exactly these column names: [{cols}].\n'

    prompt = f"""You are a data extraction expert. The user wanted to find this data on the web, but the websites were unreachable.

User query: "{query}"
{column_hint}
Using your own training knowledge, generate the most accurate and up-to-date structured data you can for this query.

Rules:
1. Return the data as a JSON array of objects where each object is one row.
2. Each object must have the same keys (column names).
3. Column names should be clean snake_case strings.
4. Include a "source" column with the value "AI Knowledge" for every row.
5. Be as accurate as possible. Use real data, not made-up examples.
6. Return 10-20 rows of data if available.
7. Return ONLY the JSON array — no explanation, no markdown, no backticks.

Example output:
[{{"name": "Python", "popularity": "28.11%", "source": "AI Knowledge"}}, {{"name": "Java", "popularity": "15.52%", "source": "AI Knowledge"}}]
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        raw = response.text.strip()
    except Exception as e:
        return {"error": f"Gemini API error: {str(e)}"}

    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Failed to generate data from AI knowledge."}

    if not isinstance(data, list) or len(data) == 0:
        return {"error": "AI could not generate relevant data for your query."}

    df = pd.DataFrame(data)
    columns = df.columns.tolist()
    rows = [sanitize_row(row) for row in df.to_dict(orient="records")]

    return {
        "success":     True,
        "query":       query,
        "columns":     columns,
        "rows":        rows,
        "total_rows":  len(rows),
        "shape":       list(df.shape),
        "sources":     [{"url": "AI Knowledge (web scraping failed)", "rows_extracted": len(rows), "pages_scraped": 0}],
        "warning":     "All discovered websites blocked or failed. Data was generated from AI knowledge instead. It may not be perfectly up-to-date.",
        "errors":      None
    }


def search_and_scrape(query: str, max_results: int = 3,
                      max_pages_per_site: int = 3) -> dict:
    """Search mode: analyze query → discover URLs → scrape with unified schema → merge."""

    # Step 1 — Analyze the query to get target schema + refined instruction
    schema = analyze_query(query)
    extraction_instruction = schema["extraction_instruction"] if schema else query
    target_columns = schema["columns"] if schema else None

    # Step 2 — Discover URLs
    url_result = discover_urls(query, max_results)
    if isinstance(url_result, dict) and "error" in url_result:
        return url_result

    urls = url_result
    all_rows = []
    sources = []
    errors = []

    # Step 3 — Scrape each discovered URL with the refined instruction + schema
    for url in urls:
        result = scrape_url(url, extraction_instruction, max_pages=max_pages_per_site)

        if result.get("error"):
            errors.append({"url": url, "error": result["error"]})
            continue

        if result.get("success") and result.get("rows"):
            site_rows = result["rows"]

            # Re-align columns to target schema if we have one
            if target_columns:
                aligned_rows = []
                for row in site_rows:
                    aligned = {}
                    for col in target_columns:
                        # Try exact match first, then fuzzy match
                        if col in row:
                            aligned[col] = row[col]
                        else:
                            # Check if any key in row contains or is contained by target col
                            matched = False
                            for key in row:
                                if col in key or key in col:
                                    aligned[col] = row[key]
                                    matched = True
                                    break
                            if not matched:
                                aligned[col] = None
                    # Only keep rows that have at least some non-null target data
                    non_null = sum(1 for v in aligned.values() if v is not None and str(v).strip())
                    if non_null >= max(1, len(target_columns) // 3):
                        aligned["source_url"] = url
                        aligned_rows.append(aligned)
                site_rows = aligned_rows
            else:
                for row in site_rows:
                    row["source_url"] = url

            if site_rows:
                all_rows.extend(site_rows)
                sources.append({
                    "url": url,
                    "rows_extracted": len(site_rows),
                    "pages_scraped": result.get("pages_scraped", 1)
                })

    # Step 4 — If all scrapes failed, fall back to AI knowledge
    if not all_rows:
        return ai_knowledge_fallback(query, schema)

    # Step 5 — Deduplicate
    seen = set()
    unique_rows = []
    for row in all_rows:
        key = frozenset(sorted((k, str(v)) for k, v in row.items()))
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)

    # Build response
    df = pd.DataFrame(unique_rows)
    columns = df.columns.tolist()
    rows = [sanitize_row(row) for row in df.to_dict(orient="records")]

    return {
        "success":      True,
        "query":        query,
        "columns":      columns,
        "rows":         rows,
        "total_rows":   len(rows),
        "shape":        list(df.shape),
        "sources":      sources,
        "errors":       errors if errors else None
    }