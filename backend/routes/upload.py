from fastapi import APIRouter, UploadFile, File
from backend.services.file_parser import parse_file

router = APIRouter()

@router.post("/")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    result = await parse_file(session_id, file)
    return result
