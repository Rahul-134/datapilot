from fastapi import APIRouter, UploadFile, File
from backend.services.file_parser import parse_file

router = APIRouter()

@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    result = await parse_file(file)
    return result