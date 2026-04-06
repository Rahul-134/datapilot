import requests
import math
import logging
import time
import pandas as pd
from bs4 import BeautifulSoup
from google import genai
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Optional Google Custom Search credentials (free 100 queries/day)
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", "").strip()
GOOGLE_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX", "").strip()

DEFAULT_MAX_PAGES = 5  # default pagination pages to follow

# ── Logging ───────────────────────────────────────────────────────────
log = logging.getLogger("scraper")
log.setLevel(logging.DEBUG)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("  [%(levelname)s] %(message)s"))
    log.addHandler(_h)

# ── Model Cascade ─────────────────────────────────────────────────────
# Ordered by preference: best quality first, then highest-quota workhorse
# NOTE: Model names must match the API exactly (use `client.models.list()` to verify)
GEMINI_MODELS = [
    "gemini-2.5-flash",              # Best quality, 20 RPD
    "gemini-3.1-flash-lite-preview", # Workhorse, 500 RPD (highest quota!)
    "gemini-3-flash-preview",        # Good quality, 20 RPD
    "gemini-2.5-flash-lite",         # Decent, 20 RPD
]

def gemini_generate(prompt: str) -> str | None:
    """Call Gemini with automatic model cascading on rate limits.
    Tries each model in GEMINI_MODELS order. On rate limit, immediately
    switches to the next model (no retry wait on same model)."""
    for model_idx, model_name in enumerate(GEMINI_MODELS):
        try:
            log.debug(f"  Trying model: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if model_idx > 0:
                log.info(f"  ✓ Succeeded with fallback model: {model_name}")
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            is_not_found = "404" in err_str or "NOT_FOUND" in err_str
            if is_rate_limit:
                log.warning(f"  {model_name} rate limited → trying next model...")
            elif is_not_found:
                log.warning(f"  {model_name} not found → trying next model...")
            else:
                log.error(f"  {model_name} error: {err_str[:150]}")
            continue  # try next model immediately

    log.error("  All Gemini models exhausted!")
    return None


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

def extract_structured_data(html: str) -> str:
    """Extract JSON-LD, OpenGraph meta tags, and tables as structured text preamble."""
    soup = BeautifulSoup(html, "lxml")
    parts = []

    # 1. JSON-LD structured data
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            ld = json.loads(script.string or "")
            parts.append(f"[STRUCTURED DATA (JSON-LD)]:\n{json.dumps(ld, indent=1, default=str)[:5000]}")
        except (json.JSONDecodeError, TypeError):
            pass

    # 2. OpenGraph / meta description
    og_tags = []
    for meta in soup.find_all("meta"):
        prop = meta.get("property", "") or meta.get("name", "")
        content = meta.get("content", "")
        if prop and content and any(k in prop.lower() for k in ["og:", "description", "author", "article:"]):
            og_tags.append(f"  {prop}: {content}")
    if og_tags:
        parts.append("[META TAGS]:\n" + "\n".join(og_tags[:15]))

    # 3. Tables → pipe-delimited text
    for i, table in enumerate(soup.find_all("table")[:5]):
        rows_text = []
        for tr in table.find_all("tr")[:100]:
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(cells):
                rows_text.append(" | ".join(cells))
        if rows_text:
            parts.append(f"[TABLE {i+1}]:\n" + "\n".join(rows_text))

    return "\n\n".join(parts)


def extract_page_links(html: str, base_url: str) -> list[dict]:
    """Extract content links that likely point to detail pages (same-domain, not nav/external)."""
    soup = BeautifulSoup(html, "lxml")
    base_domain = urlparse(base_url).netloc.lower()

    # Remove nav/footer links — we want content area links only
    for tag in soup.find_all(["nav", "footer", "header"]):
        tag.decompose()

    skip_patterns = re.compile(r"(login|signup|register|cart|checkout|privacy|terms|contact|about|faq|help|#|javascript:|mailto:)", re.I)
    links = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or skip_patterns.search(href):
            continue
        abs_url = urljoin(base_url, href)
        domain = urlparse(abs_url).netloc.lower()
        if domain != base_domain:
            continue
        if abs_url == base_url or abs_url in seen:
            continue
        text = a.get_text(strip=True)
        if len(text) < 3:
            continue
        seen.add(abs_url)
        links.append({"url": abs_url, "text": text})

    return links[:50]


def clean_html(html: str, preserve_links: bool = False) -> str:
    """Strip scripts/styles, return clean text.
    Preserves table/list structure, link hrefs (optionally), and structured data."""
    soup = BeautifulSoup(html, "lxml")

    # Remove noise — keep header/aside (they often have useful metadata)
    for tag in soup(["script", "style", "nav", "footer", "iframe", "noscript",
                      "form", "button", "svg"]):
        tag.decompose()

    # Fix truncated text
    for a_tag in soup.find_all("a", title=True):
        title = a_tag["title"].strip()
        if title:
            a_tag.string = title

    # Preserve link hrefs inline so Gemini can see detail page URLs
    if preserve_links:
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            text = a_tag.get_text(strip=True)
            if text and href and not href.startswith(("#", "javascript:")):
                a_tag.replace_with(f"{text} [LINK:{href}]")

    for img_tag in soup.find_all("img", alt=True):
        alt = img_tag["alt"].strip()
        if alt:
            img_tag.insert_after(alt)

    # Preserve table structure
    for td in soup.find_all(["td", "th"]):
        td.insert_before(" | ")
    for tr in soup.find_all("tr"):
        tr.insert_after("\n")

    # Preserve list structure
    for li in soup.find_all("li"):
        li.insert_before("• ")

    # Preserve definition lists
    for dt in soup.find_all("dt"):
        dt.insert_before("\n▸ ")
    for dd in soup.find_all("dd"):
        dd.insert_before(": ")

    text = soup.get_text(separator="\n", strip=True)

    # Smart truncation: cap at 60k chars
    max_chars = 60000
    if len(text) > max_chars:
        truncation_point = text.rfind("\n", 0, max_chars)
        if truncation_point == -1:
            truncation_point = max_chars
        text = text[:truncation_point]

    return text


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

    # Strategy 4: Numbered pagination — find current page and get next
    for selector in ["a.page-link", "a.pagination-link", ".pagination a", ".pager a"]:
        page_links = soup.select(selector)
        for i, a in enumerate(page_links):
            if a.find_parent(class_=re.compile(r"active|current|selected", re.I)):
                if i + 1 < len(page_links):
                    return urljoin(current_url, page_links[i + 1]["href"])

    # Strategy 5: URL pattern — if current URL has page=N, try page=N+1
    parsed = urlparse(current_url)
    page_match = re.search(r'[?&](page|p)=(\d+)', parsed.query)
    if page_match:
        param, num = page_match.group(1), int(page_match.group(2))
        next_num = num + 1
        next_query = re.sub(rf'{param}={num}', f'{param}={next_num}', parsed.query)
        next_url = parsed._replace(query=next_query).geturl()
        # Verify this link actually exists on the page
        for a in soup.find_all("a", href=True):
            if str(next_num) in a.get("href", ""):
                return next_url

    return None

def build_scraper_prompt(user_instruction: str, page_text: str, url: str,
                         expected_columns: list[str] | None = None,
                         structured_data: str = "") -> str:
    column_rule = ""
    if expected_columns:
        cols_str = ", ".join(f'"{c}"' for c in expected_columns)
        column_rule = (
            f"\nIMPORTANT: You MUST use exactly these column names as keys: [{cols_str}].\n"
            f"Do NOT rename, add, or remove any columns. Use these exact keys for every object.\n"
        )

    structured_section = ""
    if structured_data:
        structured_section = f"""
=== STRUCTURED DATA (HIGH PRIORITY — use this to enrich your extraction) ===
{structured_data[:8000]}
=== END STRUCTURED DATA ===
"""

    return f"""You are an elite data extraction expert. Extract the MAXIMUM amount of useful, structured data from this webpage.

URL: {url}
User Instruction: "{user_instruction}"
{column_rule}
{structured_section}
Page Content (cleaned):
{page_text}

Your job:
1. Extract ALL data matching the user's request — be EXHAUSTIVE, not selective.
2. Return the data as a JSON array of objects where each object is one row.
3. Each object must have the same keys (column names).
4. Column names should be clean snake_case strings.
5. Return ONLY the JSON array — no explanation, no markdown, no backticks.
6. If no relevant data is found, return an empty array: []

CRITICAL RULES for MAXIMUM extraction quality:
- Extract EVERY SINGLE matching item from the page — do NOT stop early or sample.
- If the page contains tabular data, extract ALL rows from ALL relevant tables.
- Values MUST be rich and detailed:
  * For descriptions: extract at least 2-3 full sentences, not truncated snippets.
  * For numeric data: include units (e.g., "$799", "4.5/5", "128GB").
  * For lists within a cell: use comma-separated format (e.g., "fever, headache, fatigue").
  * NEVER use just "N/A" or "-" if the page has ANY relevant info — extract what exists.
- If STRUCTURED DATA is provided above (JSON-LD, meta tags, tables), USE IT. It often has richer data than the page text.
- Extract SPECIFIC, ACTIONABLE data — each row = one concrete entity (product, disease, country, etc.).
- Do NOT extract: table-of-contents entries, navigation items, category headings, generic definitions, or boilerplate text.
- If the page has paragraph content about individual items, parse it into structured rows.
- Prefer COMPLETE data over more rows — 20 rows with rich detail > 50 rows with mostly empty fields.

Example output:
[{{"name": "iPhone 15", "price": "$799", "storage": "128GB/256GB/512GB", "display": "6.1-inch Super Retina XDR OLED", "chip": "A16 Bionic", "camera": "48MP main + 12MP ultrawide", "rating": "4.5/5"}}, {{"name": "Samsung S24", "price": "$799", "storage": "128GB/256GB", "display": "6.2-inch Dynamic AMOLED 2X", "chip": "Snapdragon 8 Gen 3", "camera": "50MP main + 12MP ultrawide + 10MP telephoto", "rating": "4.3/5"}}]
"""


def build_enrichment_prompt(row: dict, detail_page_text: str, detail_url: str,
                            columns: list[str]) -> str:
    """Prompt for enriching a single row with detail-page data."""
    cols_str = ", ".join(f'"{c}"' for c in columns)
    row_json = json.dumps(row, default=str)
    return f"""You are a data enrichment expert. You have a partially-filled data row extracted from a listing page.
Now you have the DETAIL PAGE for this specific item. Your job is to FILL IN missing/shallow values with rich data from the detail page.

Current row data:
{row_json}

Detail page URL: {detail_url}
Detail page content:
{detail_page_text[:15000]}

RULES:
1. Return a SINGLE JSON object with exactly these keys: [{cols_str}]
2. For fields that already have good values, KEEP THEM (don't overwrite with worse data).
3. For fields that are "N/A", empty, or very short (1-2 words), REPLACE with richer data from the detail page.
4. For description/detail fields, extract 2-4 full sentences from the detail page.
5. Return ONLY the JSON object — no explanation, no markdown, no backticks.
"""


def extract_page_data(url: str, html: str, user_instruction: str,
                      expected_columns: list[str] | None = None) -> list | dict:
    """Clean one page's HTML, extract structured data, and ask Gemini to extract structured rows."""
    # Extract structured data preamble (JSON-LD, OpenGraph, tables)
    structured = extract_structured_data(html)
    page_text = clean_html(html, preserve_links=True)

    if not page_text.strip():
        return []

    prompt = build_scraper_prompt(user_instruction, page_text, url, expected_columns, structured)

    raw = gemini_generate(prompt)
    if raw is None:
        return {"error": "Gemini API error: could not get response after retries"}

    # Strip markdown fences if model adds them
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    return data


# ── Sub-page Enrichment ───────────────────────────────────────────────

MAX_DETAIL_PAGES = 10  # max sub-pages to drill into per URL

def _fetch_and_enrich_row(row: dict, detail_url: str, columns: list[str]) -> dict | None:
    """Fetch a detail page and enrich a single row. Used in thread pool."""
    try:
        result = fetch_page(detail_url)
        if not result["success"]:
            return None
        detail_text = clean_html(result["html"])
        if not detail_text.strip():
            return None
        prompt = build_enrichment_prompt(row, detail_text, detail_url, columns)
        raw = gemini_generate(prompt)
        if raw is None:
            return None
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        enriched = json.loads(raw)
        if isinstance(enriched, dict):
            return enriched
    except Exception as e:
        log.warning(f"  Enrichment error for {detail_url}: {e}")
    return None


def enrich_rows_from_detail_pages(rows: list[dict], html: str, base_url: str,
                                  columns: list[str], max_detail: int = MAX_DETAIL_PAGES) -> list[dict]:
    """Drill into detail sub-pages to enrich rows with richer data.
    Matches listing-page links to rows by text similarity, then fetches detail pages concurrently."""
    if not rows or max_detail <= 0:
        return rows

    page_links = extract_page_links(html, base_url)
    if not page_links:
        log.info("  No detail page links found for enrichment")
        return rows

    # Try to match links to rows by finding text overlap
    # Use the first text-like column as the match key
    match_key = None
    for k in columns:
        if k not in ("source_url", "rank", "price", "rating") and rows[0].get(k):
            match_key = k
            break
    if not match_key:
        return rows

    # Build match pairs: (row_index, detail_url)
    match_pairs = []
    used_links = set()
    for i, row in enumerate(rows):
        row_val = str(row.get(match_key, "")).lower().strip()
        if not row_val or row_val == "n/a":
            continue
        best_link = None
        best_score = 0
        for link in page_links:
            if link["url"] in used_links:
                continue
            link_text = link["text"].lower().strip()
            # Check if row value appears in link text or vice versa
            if row_val in link_text or link_text in row_val:
                score = len(row_val)
            else:
                # Word overlap
                row_words = set(row_val.split())
                link_words = set(link_text.split())
                overlap = row_words & link_words
                score = len(overlap)
            if score > best_score:
                best_score = score
                best_link = link
        if best_link and best_score >= 1:
            match_pairs.append((i, best_link["url"]))
            used_links.add(best_link["url"])
        if len(match_pairs) >= max_detail:
            break

    if not match_pairs:
        log.info("  Could not match any detail page links to rows")
        return rows

    log.info(f"  Enriching {len(match_pairs)} rows from detail pages...")

    # Fetch and enrich concurrently
    enriched_map = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for row_idx, detail_url in match_pairs:
            f = executor.submit(_fetch_and_enrich_row, rows[row_idx], detail_url, columns)
            futures[f] = row_idx
            time.sleep(0.5)  # small stagger to avoid rate limits

        for future in as_completed(futures):
            row_idx = futures[future]
            try:
                enriched = future.result()
                if enriched:
                    enriched_map[row_idx] = enriched
            except Exception:
                pass

    # Merge enriched data back — only overwrite empty/shallow values
    enriched_count = 0
    for row_idx, enriched_row in enriched_map.items():
        original = rows[row_idx]
        improved = False
        for col in columns:
            old_val = str(original.get(col, "")).strip()
            new_val = str(enriched_row.get(col, "")).strip()
            # Overwrite if: old is empty/N/A OR new is significantly longer
            if new_val and new_val.lower() not in ("n/a", "none", ""):
                if not old_val or old_val.lower() in ("n/a", "none", "") or len(new_val) > len(old_val) * 1.5:
                    original[col] = enriched_row[col]
                    improved = True
        if improved:
            enriched_count += 1

    log.info(f"  Enrichment complete: {enriched_count}/{len(match_pairs)} rows improved")
    return rows


# ── Concurrent page fetching ─────────────────────────────────────────

def fetch_pages_concurrent(urls: list[str], max_workers: int = 3) -> dict[str, dict]:
    """Fetch multiple URLs concurrently. Returns {url: fetch_result}."""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(fetch_page, url): url for url in urls}
        for future in as_completed(future_map):
            url = future_map[future]
            try:
                results[url] = future.result()
            except Exception as e:
                results[url] = {"success": False, "error": str(e)}
    return results


def scrape_url(url: str, user_instruction: str, max_pages: int = DEFAULT_MAX_PAGES,
               target_columns: list[str] | None = None,
               enable_enrichment: bool = True) -> dict:
    all_rows = []
    current_url = url
    pages_fetched = 0
    ran_out_of_pages = False
    first_page_html = None

    while current_url and pages_fetched < max_pages:
        # Fetch page
        fetch_result = fetch_page(current_url)
        if not fetch_result["success"]:
            if pages_fetched == 0:
                return {"error": fetch_result["error"]}
            break  # stop pagination on error, keep what we have

        html = fetch_result["html"]
        pages_fetched += 1

        # Save first page HTML for detail-page enrichment later
        if pages_fetched == 1:
            first_page_html = html

        # On the first page, use target_columns if provided (search mode);
        # on subsequent pages, enforce columns from page 1's data.
        if all_rows:
            expected_cols = list(all_rows[0].keys())
        elif target_columns:
            expected_cols = target_columns
        else:
            expected_cols = None
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

    # Sub-page detail enrichment (the big quality uplift)
    if enable_enrichment and first_page_html and all_rows:
        columns = list(all_rows[0].keys())
        all_rows = enrich_rows_from_detail_pages(all_rows, first_page_html, url, columns)

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

def google_search(query: str, num_results: int = 5) -> list[dict] | None:
    """Use Google Custom Search JSON API to find real URLs.
    Returns a list of {url, title, snippet} dicts, or None if not configured."""
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_CX:
        return None  # not configured, caller should use fallback

    blocked_domains = {"wikipedia.org"}

    try:
        # Append -wikipedia to bias results away from Wikipedia
        search_q = f"{query} -wikipedia" if "wikipedia" not in query.lower() else query
        log.info(f"Google Search API: {search_q}")
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": GOOGLE_SEARCH_API_KEY,
                "cx": GOOGLE_SEARCH_CX,
                "q": search_q,
                "num": min(num_results, 10),
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("items", []):
            url = item["link"]
            domain = urlparse(url).netloc.lower()
            # Skip Wikipedia results
            if any(b in domain for b in blocked_domains):
                continue
            results.append({
                "url": url,
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
            })
        log.info(f"Google Search API: {len(results)} results (Wikipedia filtered)")
        for r in results:
            log.info(f"  → {r['url']}")
        return results if results else None

    except Exception as e:
        log.warning(f"Google Search API error: {e}")
        return None  # fall back to other methods


def duckduckgo_search(query: str, max_results: int = 8) -> list[str]:
    """Search DuckDuckGo HTML (no API key needed) and extract result URLs.
    This is the free, zero-config fallback for URL discovery."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    try:
        log.info(f"DuckDuckGo: {query}")
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        urls = []
        # DuckDuckGo HTML results have class "result__a" for links
        for a_tag in soup.select("a.result__a"):
            href = a_tag.get("href", "")
            if href.startswith("http"):
                urls.append(href)
            elif href.startswith("//duckduckgo.com/l/"):
                # DDG redirect links — extract actual URL from uddg param
                from urllib.parse import parse_qs, urlparse as _urlparse
                parsed = _urlparse(href)
                params = parse_qs(parsed.query)
                if "uddg" in params:
                    urls.append(params["uddg"][0])

        log.info(f"DuckDuckGo: {len(urls)} result URLs")
        return urls[:max_results]

    except Exception as e:
        log.warning(f"DuckDuckGo error: {e}")
        return []


def discover_urls_via_search(search_queries: list[str], max_results: int = 5) -> list[str]:
    """Use Google Custom Search API with multiple search queries for diversity.
    Returns a deduplicated list of URLs from different domains."""
    all_results = []
    seen_domains = set()

    for query in search_queries:
        results = google_search(query, num_results=max_results)
        if not results:
            continue

        for r in results:
            domain = urlparse(r["url"]).netloc.lower()
            # Skip duplicate domains for diversity
            if domain in seen_domains:
                continue
            seen_domains.add(domain)
            all_results.append(r["url"])

    return all_results[:max_results] if all_results else []


def _url_relevance_score(url: str, query: str) -> float:
    """Score how relevant a URL is to the query (0.0 = irrelevant, 1.0 = perfect).
    Uses keyword overlap between query words and URL domain+path."""
    # Extract keywords from query (lowercase, 3+ chars, no stopwords)
    stopwords = {"the", "and", "for", "from", "with", "that", "this", "all", "are",
                 "was", "were", "been", "have", "has", "had", "like", "data", "list",
                 "table", "year", "get", "find", "show", "about"}
    query_words = set()
    for w in re.split(r'\W+', query.lower()):
        if len(w) >= 3 and w not in stopwords:
            query_words.add(w)

    if not query_words:
        return 0.5  # can't determine relevance

    parsed = urlparse(url)
    # Combine domain + path for matching
    url_text = (parsed.netloc + " " + parsed.path).lower().replace("-", " ").replace("_", " ").replace("/", " ")

    # Count keyword matches
    matches = sum(1 for w in query_words if w in url_text)
    # Also check partial matches (e.g., "ipl" in "ipl-results")
    partial = sum(0.5 for w in query_words if any(w in part for part in url_text.split()) and w not in url_text.split())

    score = (matches + partial) / len(query_words)
    return min(score, 1.0)


def discover_urls_via_duckduckgo(search_queries: list[str], query: str,
                                  max_results: int = 5) -> list[str]:
    """Use DuckDuckGo HTML search (no API key needed) for URL discovery.
    Filters out Wikipedia, enforces domain diversity, and validates URL relevance."""
    blocked_domains = {"wikipedia.org", "en.wikipedia.org", "simple.wikipedia.org",
                       "www.wikipedia.org", "en.m.wikipedia.org",
                       "academia.edu", "researchgate.net", "quora.com"}
    all_urls = []
    seen_domains = set()

    queries_to_try = (search_queries or []) + [query]
    # Only add "-wikipedia" to first 2 queries — it can hurt niche searches
    processed = []
    for i, q in enumerate(queries_to_try):
        if i < 2 and "wikipedia" not in q.lower():
            processed.append(f"{q} -wikipedia")
        else:
            processed.append(q)
    queries_to_try = list(dict.fromkeys(processed))  # deduplicate

    for q in queries_to_try[:5]:  # try up to 5 queries
        urls = duckduckgo_search(q, max_results=10)
        for url in urls:
            domain = urlparse(url).netloc.lower().replace("www.", "")
            # Skip blocked domains
            if any(blocked in domain for blocked in blocked_domains):
                continue
            # Skip duplicate domains
            base_domain = ".".join(domain.split(".")[-2:])
            if base_domain in seen_domains:
                continue

            # Relevance check — skip URLs that have zero relation to the query
            relevance = _url_relevance_score(url, query)
            if relevance < 0.1:
                log.debug(f"  DDG skip (irrelevant, score={relevance:.2f}): {url}")
                continue

            seen_domains.add(base_domain)
            all_urls.append(url)

            if len(all_urls) >= max_results:
                break
        if len(all_urls) >= max_results:
            break

    log.info(f"DuckDuckGo Discovery: {len(all_urls)} relevant URLs found")
    for u in all_urls:
        log.info(f"  → {u}")
    return all_urls[:max_results]


def discover_urls_via_gemini(query: str, search_queries: list[str] | None,
                              max_results: int = 3) -> list[str] | dict:
    """Last-resort fallback: ask Gemini to suggest URLs.
    This version bans Wikipedia, demands diversity, and emphasizes
    sites known to serve HTML content to simple HTTP clients."""

    search_hints = ""
    if search_queries:
        hints = "\n".join(f"  - {q}" for q in search_queries)
        search_hints = f"\nHere are some search queries that might help you think of relevant sites:\n{hints}\n"

    prompt = f"""You are a web research assistant. The user wants to scrape structured data from the internet using a simple HTTP GET request.

User query: "{query}"
{search_hints}
Your job: Suggest up to {max_results} real, publicly accessible website URLs that contain the SPECIFIC data the user described.

STRICT RULES:
1. **NEVER suggest Wikipedia** (any *.wikipedia.org domain). Wikipedia has generic definitions, NOT useful structured data.
2. **NEVER suggest the same domain twice.** Every URL must be from a DIFFERENT website.
3. URLs must serve their data as plain HTML that a simple HTTP GET receives — NO JavaScript-rendered SPAs.
4. Each URL must point to a SPECIFIC page with actual data — NOT a homepage.
5. **CRITICALLY IMPORTANT**: Only suggest sites where data is in the raw HTML, not loaded via JavaScript. Good indicators:
   - The site is older/simpler in design
   - The site uses server-side rendering
   - The URL path suggests a specific data page (e.g., /conditions/list, /diseases-a-z)
6. PROVEN SCRAPABLE sites for different categories:
   - MEDICAL: drugs.com/condition/, rxlist.com, medicinenet.com, patient.info, emedicinehealth.com
   - FINANCE: worldometers.info, tradingeconomics.com, xe.com, goodreturns.in
   - TECH: w3techs.com, db-engines.com/en/ranking, tiobe.com/tiobe-index
   - GENERAL: worldpopulationreview.com, worldometers.info, github.com
   - BOOKS: openlibrary.org, goodreads.com/list, books.toscrape.com
7. AVOID sites that block simple HTTP scraping:
   - webmd.com, healthline.com, mayoclinic.org, clevelandclinic.org (all JS-heavy)
   - statista.com, bloomberg.com, imf.org (paywalls/CAPTCHAs)
   - Any site returning mostly empty HTML with JS loaders

Return ONLY a JSON array of URL strings — no explanation, no markdown, no backticks.
"""
    print(f"  [Gemini URL Discovery] Asking Gemini for URLs...")
    raw = gemini_generate(prompt)
    if raw is None:
        return {"error": "Gemini API error while discovering URLs (rate limit or other error)"}

    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        urls = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Failed to parse URL suggestions from the AI model."}

    if not isinstance(urls, list):
        return {"error": "AI model returned unexpected format for URL suggestions."}

    # Filter to valid URLs and enforce no Wikipedia + domain diversity
    blocked_domains = {"wikipedia.org"}
    seen_domains = set()
    filtered = []

    for u in urls:
        if not isinstance(u, str) or not u.startswith(("http://", "https://")):
            continue
        domain = urlparse(u).netloc.lower().replace("www.", "")
        # Skip Wikipedia
        if any(blocked in domain for blocked in blocked_domains):
            continue
        # Skip duplicate domains
        base_domain = ".".join(domain.split(".")[-2:])
        if base_domain in seen_domains:
            continue
        seen_domains.add(base_domain)
        filtered.append(u)

    print(f"  [Gemini URL Discovery] Suggested URLs: {filtered}")

    if not filtered:
        return {"error": "The AI model could not find relevant non-Wikipedia websites for your query. Try rephrasing."}

    return filtered[:max_results]


def discover_urls(query: str, search_queries: list[str] | None = None,
                  max_results: int = 5) -> list[str] | dict:
    """Discover URLs using a cascade: Google Search API → DuckDuckGo → Gemini.
    Uses multiple search queries for source diversity."""
    log.info(f"{'='*50}")
    log.info(f"URL Discovery for: {query[:80]}")
    log.info(f"Search queries: {search_queries}")
    log.info(f"{'='*50}")

    # Try 1: Google Custom Search API (best quality, needs API key)
    if GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX:
        log.info("Trying Google Custom Search API...")
        if search_queries:
            urls = discover_urls_via_search(search_queries, max_results)
            if urls:
                log.info(f"✓ Google API (multi-query) → {len(urls)} URLs")
                return urls

        results = google_search(query, max_results)
        if results:
            seen = set()
            urls = []
            for r in results:
                domain = urlparse(r["url"]).netloc.lower()
                if domain not in seen:
                    seen.add(domain)
                    urls.append(r["url"])
            if urls:
                log.info(f"✓ Google API (single query) → {len(urls)} URLs")
                return urls[:max_results]

    # Try 2: DuckDuckGo HTML search (free, no API key needed)
    log.info("Trying DuckDuckGo HTML search...")
    ddg_urls = discover_urls_via_duckduckgo(search_queries or [], query, max_results)
    if len(ddg_urls) >= max_results:
        log.info(f"✓ DuckDuckGo → {len(ddg_urls)} URLs")
        return ddg_urls

    # Try 3: Gemini-based URL discovery
    # If DDG found some but not enough, combine with Gemini results
    log.info("Trying Gemini-based URL discovery...")
    gemini_result = discover_urls_via_gemini(query, search_queries, max_results)

    if isinstance(gemini_result, dict) and "error" in gemini_result:
        # Gemini failed — return DDG results if we have any, otherwise the error
        if ddg_urls:
            log.info(f"Gemini failed, using {len(ddg_urls)} DDG URL(s)")
            return ddg_urls
        return gemini_result

    # Combine DDG + Gemini, dedup by domain
    combined = list(ddg_urls)
    seen = {urlparse(u).netloc.lower() for u in combined}
    for u in gemini_result:
        domain = urlparse(u).netloc.lower()
        if domain not in seen:
            seen.add(domain)
            combined.append(u)
    log.info(f"✓ Combined DDG({len(ddg_urls)}) + Gemini({len(gemini_result)}) → {len(combined)} URLs")
    return combined[:max_results]


def analyze_query(query: str) -> dict:
    """Analyze the user's query to determine target schema, refined extraction instruction,
    and optimized search queries for diverse URL discovery."""
    prompt = f"""You are a data analysis and web research expert. A user wants to search the web and scrape structured data.

User query: "{query}"

You must do THREE things:

## 1. Understand the Data Need
Deeply analyze what the user ACTUALLY wants. Think about:
- What specific entities/items should each row represent?
- What attributes/properties are most useful for each entity?
- What level of detail is needed — aim for COMPREHENSIVE, RICH data
- What would make this data genuinely useful for analysis, training, or decision-making?

## 2. Define the Schema
Create a COMPREHENSIVE schema that captures the MAXIMUM useful information.
- Use descriptive snake_case column names
- Include 8-12 columns that capture rich, multi-dimensional data
- Always include at least one DESCRIPTION or DETAILS column with 2+ sentences expected
- Think about what makes data genuinely USEFUL — not just names and numbers, but context, details, relationships
- Consider columns like: descriptions, categories, dates, metrics, relationships, notable_features

## 3. Generate Search Queries
Create 4 different Google search queries that will find DIVERSE, DATA-RICH pages.
- Each query should target a DIFFERENT type of source
- Include site-specific operators where helpful
- Focus on pages that contain STRUCTURED data (tables, lists, databases)
- At least one query should target a site known to have rich detail pages (for sub-page drilling)

Return a JSON object with exactly these keys:
- "columns": array of snake_case column names (8-12 columns)
- "extraction_instruction": a very specific, detailed instruction for extracting data from any webpage. For EACH column, explain what data should go in it and how detailed it should be. Be explicit about minimum detail levels (e.g., "descriptions should be 2-3 sentences"). Mention the kind of data that should fill each column.
- "search_queries": array of 4 different Google search query strings for finding diverse sources
- "detail_worthy": boolean — true if this data type typically benefits from drilling into individual item pages for richer details (e.g., products, diseases, companies = true; simple rankings or statistics = false)

Rules:
- Do NOT include generic metadata columns like "source_url" — those are added automatically.
- Focus columns on the actual data the user wants, not meta-information.
- The extraction instruction should be specific enough that data from different websites will have the same structure.
- The extraction instruction MUST include: "Extract EVERY matching item exhaustively. Do NOT extract generic category definitions, table-of-contents entries, or navigation text. Values must be detailed — descriptions of at least 2 sentences, lists as comma-separated items."

Return ONLY the JSON object — no explanation, no markdown, no backticks.

Example for query "Data to train a medical chatbot that reads symptoms and tells diseases and cures":
{{"columns": ["disease_name", "category", "common_symptoms", "causes_risk_factors", "diagnosis_methods", "treatment_options", "medications", "prevention", "when_to_see_doctor", "severity_level", "description"], "extraction_instruction": "Extract a list of specific diseases or medical conditions. For each: disease_name = exact medical name; category = type (infectious, chronic, autoimmune, etc.); common_symptoms = comma-separated list of 5+ symptoms; causes_risk_factors = known causes and risk factors (2+ sentences); diagnosis_methods = how it's diagnosed; treatment_options = all treatments (medications, procedures, home remedies); medications = specific drug names if available; prevention = preventive measures; when_to_see_doctor = warning signs requiring medical attention; severity_level = mild/moderate/severe/life-threatening; description = 2-3 sentence overview. Only include SPECIFIC diseases (e.g., 'Influenza', 'Type 2 Diabetes'). Do NOT extract generic categories. Extract EVERY matching item exhaustively.", "search_queries": ["common diseases symptoms causes treatment comprehensive list", "diseases A-Z symptoms diagnosis treatment database", "medical conditions medications side effects drug interactions", "disease symptoms treatment prevention guide comprehensive"], "detail_worthy": true}}
"""
    raw = gemini_generate(prompt)
    if raw is None:
        return None

    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if isinstance(result, dict) and "columns" in result and "extraction_instruction" in result:
        # Ensure search_queries exists
        if "search_queries" not in result or not isinstance(result["search_queries"], list):
            result["search_queries"] = [query]  # fallback to raw query
        # Default detail_worthy to True
        if "detail_worthy" not in result:
            result["detail_worthy"] = True
        return result
    return None


def filter_irrelevant_rows(rows: list[dict], query: str, columns: list[str]) -> list[dict]:
    """Use a quick Gemini call to do ROW-LEVEL filtering instead of all-or-nothing.
    Keeps rows that are relevant and discards only truly irrelevant ones."""
    if not rows or len(rows) <= 2:
        return rows  # too few to filter

    # For large batches, work on a representative sample to decide
    # whether to keep the batch, then do row-level filtering only if mixed
    sample_size = min(len(rows), 10)
    sample = rows[:sample_size]
    sample_json = json.dumps(sample, indent=2, default=str)

    prompt = f"""You are a data quality expert. A user searched for:
"{query}"

Here are {sample_size} rows from the extracted data (0-indexed):
{sample_json}

For EACH row, decide if it is RELEVANT to the user's query.

A row is RELEVANT if it contains specific, actionable information matching the query:
- Specific items (diseases, products, languages, etc.) with concrete details → RELEVANT
- Generic definitions, category headings, navigation text, boilerplate → NOT RELEVANT
- Partially useful data (e.g., a disease name with some missing fields) → RELEVANT (partial data is still useful)

Be GENEROUS — when in doubt, mark as relevant. It's better to keep slightly noisy data than to lose good data.

Return a JSON object with:
- "dominated_by_relevant": true if MOST rows (>50%) are relevant, false if most are junk
- "irrelevant_indices": array of 0-based indices of rows that are clearly NOT relevant (empty array if all are relevant)
- "reason": short explanation

Return ONLY the JSON object — no explanation, no markdown, no backticks.
"""
    try:
        raw = gemini_generate(prompt)
        if raw is None:
            log.warning("  Filter: Gemini call failed, keeping all rows")
            return rows
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        result = json.loads(raw)

        if not isinstance(result, dict):
            log.warning("  Filter returned non-dict, keeping all rows")
            return rows

        log.info(f"  Filter result: dominated_by_relevant={result.get('dominated_by_relevant')}, "
                 f"irrelevant_indices={result.get('irrelevant_indices', [])}, "
                 f"reason={result.get('reason', 'N/A')}")

        # If the batch is dominated by relevant data, do surgical removal
        if result.get("dominated_by_relevant", True):
            bad_indices = set(result.get("irrelevant_indices", []))
            if bad_indices and len(bad_indices) < len(rows):
                # Only remove bad rows from the sample range; keep the rest
                filtered = [row for i, row in enumerate(rows) if i not in bad_indices]
                log.info(f"  Surgical filter: removed {len(bad_indices)} irrelevant rows, kept {len(filtered)}")
                return filtered
            return rows  # all good, keep everything
        else:
            # Most rows are junk — but still check if ANY are worth keeping
            bad_indices = set(result.get("irrelevant_indices", []))
            if bad_indices and len(bad_indices) < len(rows):
                # Keep whatever is NOT in the bad list
                filtered = [row for i, row in enumerate(rows) if i not in bad_indices]
                if filtered:
                    log.info(f"  Partial salvage: kept {len(filtered)} relevant rows from mostly-junk batch")
                    return filtered
            # Even when "dominated_by_relevant" is false, keep rows if we can't
            # identify specific bad ones — losing data is worse than noisy data
            log.warning(f"  Filter says batch is mostly junk but no specific indices — keeping all {len(rows)} rows")
            return rows

    except Exception as e:
        log.warning(f"  Filter error ({e}), keeping all rows")

    return rows


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
6. Return 15-30 rows of data if available.
7. Each row must represent a SPECIFIC, CONCRETE item — not a generic category or definition.
8. Return ONLY the JSON array — no explanation, no markdown, no backticks.

Example output:
[{{"name": "Python", "popularity": "28.11%", "source": "AI Knowledge"}}, {{"name": "Java", "popularity": "15.52%", "source": "AI Knowledge"}}]
"""
    raw = gemini_generate(prompt)
    if raw is None:
        return {"error": "Gemini API error: could not generate data after retries"}

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


MIN_USEFUL_ROWS = 10  # if web scraping yields fewer rows, supplement with AI knowledge

def search_and_scrape(query: str, max_results: int = 5,
                      max_pages_per_site: int = 3) -> dict:
    """Search mode: analyze query → discover URLs → scrape with unified schema → filter → merge.
    If scraping yields too few rows, supplements with AI-generated data."""
    log.info(f"{'='*60}")
    log.info(f"SEARCH & SCRAPE: {query}")
    log.info(f"{'='*60}")

    # Step 1 — Analyze the query to get target schema + refined instruction + search queries
    log.info("Step 1: Analyzing query...")
    schema = analyze_query(query)
    if schema:
        extraction_instruction = schema["extraction_instruction"]
        target_columns = schema["columns"]
        search_queries = schema.get("search_queries")
        detail_worthy = schema.get("detail_worthy", True)
    else:
        # Fallback: generate simple search queries from the raw query
        log.warning("  analyze_query failed (API issue?), using raw query as fallback")
        extraction_instruction = query
        target_columns = None
        detail_worthy = False
        # Generate basic search queries from the user's query
        search_queries = [
            f"{query} data table list",
            f"{query} database",
            query,
        ]
    log.info(f"  Columns: {target_columns}")
    log.info(f"  Detail-worthy: {detail_worthy}")
    log.info(f"  Search queries: {search_queries}")
    log.info(f"  Extraction: {extraction_instruction[:120]}...")

    # Step 2 — Discover URLs (Google API → DuckDuckGo → Gemini)
    log.info("Step 2: Discovering URLs...")
    url_result = discover_urls(query, search_queries, max_results)
    if isinstance(url_result, dict) and "error" in url_result:
        log.warning(f"URL discovery failed: {url_result['error']}")
        # Try AI fallback, but if that also fails, return a user-friendly error
        ai_result = ai_knowledge_fallback(query, schema)
        if ai_result.get("error"):
            return {"error": "Could not discover URLs or generate data. The AI API may be temporarily rate-limited. Please try again in a few minutes."} 
        return ai_result

    urls = url_result
    log.info(f"  Found {len(urls)} URLs to scrape")
    all_rows = []
    sources = []
    errors = []

    # Step 3 — Scrape each discovered URL with the refined instruction + schema
    for i, url in enumerate(urls, 1):
        log.info(f"Step 3: Scraping URL {i}/{len(urls)}: {url}")

        # Small delay between URLs to avoid Gemini rate limits
        if i > 1:
            time.sleep(2)

        result = scrape_url(url, extraction_instruction,
                            max_pages=max_pages_per_site,
                            target_columns=target_columns,
                            enable_enrichment=detail_worthy)

        if result.get("error"):
            log.warning(f"  ✗ Error scraping {url}: {result['error']}")
            errors.append({"url": url, "error": result["error"]})
            continue

        if result.get("success") and result.get("rows"):
            site_rows = result["rows"]
            log.info(f"  ✓ Extracted {len(site_rows)} rows")

            # Add source_url to each row
            for row in site_rows:
                row["source_url"] = url

            # Step 3.5 — Relevance filter (skip to save API quota if we have few URLs)
            # Only filter if we have more than 3 URLs being scraped
            if site_rows and len(urls) <= 3:
                log.info(f"  Skipping relevance filter (only {len(urls)} URLs, saving API quota)")
            elif site_rows:
                before = len(site_rows)
                site_rows = filter_irrelevant_rows(site_rows, query, target_columns or [])
                log.info(f"  Relevance filter: {before} → {len(site_rows)} rows")

            if site_rows:
                all_rows.extend(site_rows)
                sources.append({
                    "url": url,
                    "rows_extracted": len(site_rows),
                    "pages_scraped": result.get("pages_scraped", 1)
                })

    log.info(f"Step 4: Total scraped rows = {len(all_rows)} from {len(sources)} sources")

    # Step 4 — Supplement with AI knowledge if we have too few rows
    ai_supplemented = False
    web_row_count = len(all_rows)
    if web_row_count < MIN_USEFUL_ROWS:
        needed = MIN_USEFUL_ROWS - web_row_count
        log.info(f"  Only {web_row_count} rows scraped (< {MIN_USEFUL_ROWS}). "
                 f"Supplementing with ~{needed}+ rows from AI knowledge...")

        ai_result = ai_knowledge_fallback(query, schema)
        if ai_result.get("success") and ai_result.get("rows"):
            ai_rows = ai_result["rows"]

            # Harmonize AI rows with scraped data columns
            # Remove 'source' column (AI fallback adds it), use 'source_url' instead
            for row in ai_rows:
                row.pop("source", None)  # remove 'source' if present
                row["source_url"] = "AI Knowledge"

            # If we already have some scraped data, align AI columns to match
            if all_rows:
                scraped_keys = set(all_rows[0].keys())
                harmonized_ai_rows = []
                for row in ai_rows:
                    harmonized = {}
                    for key in scraped_keys:
                        harmonized[key] = row.get(key, "N/A")
                    harmonized["source_url"] = "AI Knowledge"
                    harmonized_ai_rows.append(harmonized)
                ai_rows = harmonized_ai_rows

            # Deduplicate AI rows against existing scraped data
            # Use the first non-source column as a dedup key
            existing_values = set()
            dedup_key = None
            if all_rows:
                for k in all_rows[0].keys():
                    if k != "source_url":
                        dedup_key = k
                        break
                if dedup_key:
                    existing_values = {str(r.get(dedup_key, "")).lower().strip() for r in all_rows}

            unique_ai_rows = []
            for row in ai_rows:
                if dedup_key:
                    val = str(row.get(dedup_key, "")).lower().strip()
                    if val in existing_values:
                        continue
                    existing_values.add(val)
                unique_ai_rows.append(row)

            if unique_ai_rows:
                all_rows.extend(unique_ai_rows)
                sources.append({
                    "url": "AI Knowledge (supplement)",
                    "rows_extracted": len(unique_ai_rows),
                    "pages_scraped": 0
                })
                ai_supplemented = True
                log.info(f"  Added {len(unique_ai_rows)} AI-generated rows (deduplicated from {len(ai_rows)})")

    # If still no rows at all, pure AI fallback
    if not all_rows:
        log.warning("No data at all — full AI knowledge fallback")
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

    result = {
        "success":      True,
        "query":        query,
        "columns":      columns,
        "rows":         rows,
        "total_rows":   len(rows),
        "shape":        list(df.shape),
        "sources":      sources,
        "errors":       errors if errors else None
    }

    if ai_supplemented:
        web_row_count = sum(s["rows_extracted"] for s in sources if "AI Knowledge" not in s["url"])
        result["info"] = (
            f"Scraped {web_row_count} rows from the web and supplemented with "
            f"AI-generated data to provide comprehensive results."
        )

    log.info(f"DONE: {len(rows)} total rows from {len(sources)} sources")
    return result