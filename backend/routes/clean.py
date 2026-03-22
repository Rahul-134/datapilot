from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.file_parser import get_stored_df
from backend.services.data_cleaner import clean_dataframe

router = APIRouter()

class CleanRequest(BaseModel):
    remove_duplicates: bool = True
    handle_nulls: bool = True
    null_strategy: str = "drop"   # drop | mean | median | unknown

@router.post("/")
def clean_data(req: CleanRequest):
    df = get_stored_df()

    if df is None:
        return {"error": "No file uploaded yet. Please upload a file first."}

    result = clean_dataframe(
        df.copy(),                  # always clean a copy, keep original safe
        req.remove_duplicates,
        req.handle_nulls,
        req.null_strategy
    )

    return result