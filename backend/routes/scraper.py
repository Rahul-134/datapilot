from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from backend.services.scraper_service import scrape_url
import pandas as pd
import io

router = APIRouter()

# Store last scrape result for download
_scrape_store: dict = {"df": None}

class ScrapeRequest(BaseModel):
    url:         str
    instruction: str
    max_pages:   int = Field(default=5, ge=1, le=20)

@router.post("/")
def run_scrape(req: ScrapeRequest):
    if not req.url.strip():
        return {"error": "URL cannot be empty."}
    if not req.instruction.strip():
        return {"error": "Instruction cannot be empty."}

    # Basic URL validation
    if not req.url.startswith(("http://", "https://")):
        return {"error": "URL must start with http:// or https://"}

    result = scrape_url(req.url.strip(), req.instruction.strip(), req.max_pages)

    if result.get("success"):
        _scrape_store["df"] = pd.DataFrame(result["rows"], columns=result["columns"])

    return result

@router.get("/download")
def download_scrape(format: str = "csv"):
    df = _scrape_store["df"]
    if df is None:
        return {"error": "No scrape result available. Run a scrape first."}

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