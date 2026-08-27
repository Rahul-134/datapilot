from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from backend.services.scraper_service import scrape_url, search_and_scrape
from backend.services.session_store import get_value, set_value
import pandas as pd
import io

router = APIRouter()

class ScrapeRequest(BaseModel):
    url:         str
    instruction: str
    max_pages:   int = Field(default=5, ge=1, le=20)

class SearchScrapeRequest(BaseModel):
    query:              str
    max_results:        int = Field(default=3, ge=1, le=10)
    max_pages_per_site: int = Field(default=3, ge=1, le=10)

@router.post("/")
def run_scrape(session_id: str, req: ScrapeRequest):
    if not req.url.strip():
        return {"error": "URL cannot be empty."}
    if not req.instruction.strip():
        return {"error": "Instruction cannot be empty."}

    # Basic URL validation
    if not req.url.startswith(("http://", "https://")):
        return {"error": "URL must start with http:// or https://"}

    result = scrape_url(req.url.strip(), req.instruction.strip(), req.max_pages)

    if result.get("success"):
        result_df = pd.DataFrame(result["rows"], columns=result["columns"])
        set_value(session_id, "scrape_df", result_df.to_csv(index=False))

    return result

@router.post("/search")
def run_search_scrape(session_id: str, req: SearchScrapeRequest):
    if not req.query.strip():
        return {"error": "Search query cannot be empty."}

    result = search_and_scrape(req.query.strip(), req.max_results, req.max_pages_per_site)

    if result.get("success"):
        result_df = pd.DataFrame(result["rows"], columns=result["columns"])
        set_value(session_id, "scrape_df", result_df.to_csv(index=False))

    return result

@router.get("/download")
def download_scrape(session_id: str, format: str = "csv"):
    csv_text = get_value(session_id, "scrape_df")
    if csv_text is None:
        return {"error": "No scrape result available. Run a scrape first."}

    df = pd.read_csv(io.StringIO(csv_text))

    buf      = io.BytesIO()
    out_name = "scraped_data"

    if format == "xlsx":
        df.to_excel(buf, index=False)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        out_name  += ".xlsx"
    else:
        df.to_csv(buf, index=False)
        media_type = "text/csv"
        out_name  += ".csv"

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={out_name}"}
    )
