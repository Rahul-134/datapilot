from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.services.file_parser import get_stored_df
from backend.services.llm_service import query_dataframe, build_prompt, execute_code
from backend.services.session_store import get_value, set_value
import pandas as pd
import io

router = APIRouter()

class QueryRequest(BaseModel):
    prompt: str

@router.post("/")
def run_query(session_id: str, req: QueryRequest):
    df = get_stored_df(session_id)

    if df is None:
        return {"error": "No file uploaded yet. Please upload a file first."}

    if not req.prompt.strip():
        return {"error": "Prompt cannot be empty."}

    result = query_dataframe(req.prompt.strip(), df)

    if result.get("success"):
        # Store result df for download
        result_df = pd.DataFrame(result["rows"], columns=result["columns"])
        set_value(session_id, "query_df", result_df.to_csv(index=False))
        set_value(session_id, "query_prompt", req.prompt.strip())

    return result

@router.get("/download")
def download_query_result(session_id: str, format: str = "csv"):
    csv_text = get_value(session_id, "query_df")

    if csv_text is None:
        return {"error": "No query result available. Run a query first."}

    df = pd.read_csv(io.StringIO(csv_text))

    buf      = io.BytesIO()
    out_name = "query_result"

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
