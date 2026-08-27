from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.services.file_parser import get_stored_df, get_stored_filename, set_stored_df
from backend.services.data_cleaner import clean_dataframe
from backend.services.session_store import get_value, set_value, clear_session
import pandas as pd
import io

router = APIRouter()

class CleanRequest(BaseModel):
    remove_duplicates: bool = True
    handle_nulls: bool = True
    null_strategy: str = "drop"

@router.post("/")
def clean_data(session_id: str, req: CleanRequest):
    df       = get_stored_df(session_id)
    filename = get_stored_filename(session_id)

    if df is None:
        return {"error": "No file uploaded yet. Please upload a file first."}

    result = clean_dataframe(
        df.copy(),
        req.remove_duplicates,
        req.handle_nulls,
        req.null_strategy
    )

    # ✅ Update active df so queries now run on cleaned data
    set_stored_df(session_id, result["_df"])

    # Store for download
    set_value(session_id, "cleaned_df", result["_df"].to_csv(index=False))
    set_value(session_id, "cleaned_filename", filename or "")

    # Strip internal keys before returning
    result.pop("_df")
    result.pop("filename")

    return result

@router.get("/download")
def download_clean(session_id: str):
    csv_text = get_value(session_id, "cleaned_df")
    filename = get_value(session_id, "cleaned_filename")

    if csv_text is None:
        return {"error": "No cleaned data available. Please clean a file first."}

    df = pd.read_csv(io.StringIO(csv_text))

    base     = filename.rsplit(".", 1)[0] if filename else "data"
    ext      = filename.rsplit(".", 1)[-1] if filename else "csv"
    out_name = f"{base}_cleaned.{ext}"

    buf = io.BytesIO()
    if ext in ("xlsx", "xls"):
        df.to_excel(buf, index=False)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        df.to_csv(buf, index=False)
        media_type = "text/csv"

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={out_name}"}
    )

@router.post("/reset")
def reset_data(session_id: str):
    clear_session(session_id)
    return {"success": True, "message": "Session reset."}
