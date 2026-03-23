from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.file_parser import get_stored_df
from backend.services.llm_service import query_dataframe

router = APIRouter()

class QueryRequest(BaseModel):
    prompt: str

@router.post("/")
def run_query(req: QueryRequest):
    df = get_stored_df()

    if df is None:
        return {"error": "No file uploaded yet. Please upload a file first."}

    if not req.prompt.strip():
        return {"error": "Prompt cannot be empty."}

    result = query_dataframe(req.prompt.strip(), df)
    return result