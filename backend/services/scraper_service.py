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

MAX_PAGES = 5  # max pagination pages to follow

def sanitize(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value

def sanitize_row(row: dict) -> dict:
    return {k: sanitize(v) for k, v in row.items()}

def fetch_page(url: str) -> dict:
    """Fetch raw HTML from a URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
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

def build_scraper_prompt(user_instruction: str, page_text: str, url: str) -> str:
    return f"""You are a data extraction expert. A user wants to extract structured data from a webpage.

URL: {url}
User Instruction: "{user_instruction}"

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

def extract_page_data(url: str, html: str, user_instruction: str) -> list | dict:
    """Clean one page's HTML and ask Gemini to extract structured data."""
    page_text = clean_html(html)

    if not page_text.strip():
        return []

    prompt = build_scraper_prompt(user_instruction, page_text, url)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
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

def scrape_url(url: str, user_instruction: str) -> dict:
    all_rows = []
    current_url = url
    pages_fetched = 0

    while current_url and pages_fetched < MAX_PAGES:
        # Fetch page
        fetch_result = fetch_page(current_url)
        if not fetch_result["success"]:
            if pages_fetched == 0:
                return {"error": fetch_result["error"]}
            break  # stop pagination on error, keep what we have

        html = fetch_result["html"]
        pages_fetched += 1

        # Extract data from this page
        page_data = extract_page_data(current_url, html, user_instruction)

        if isinstance(page_data, dict) and "error" in page_data:
            if pages_fetched == 1:
                return page_data
            break

        all_rows.extend(page_data)

        # Check for next page
        current_url = detect_next_page(html, current_url)

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

    return {
        "success":      True,
        "url":          url,
        "instruction":  user_instruction,
        "columns":      columns,
        "rows":         rows,
        "total_rows":   len(df),
        "shape":        list(df.shape),
        "pages_scraped": pages_fetched
    }