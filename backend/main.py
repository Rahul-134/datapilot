import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

ML_TRAINER_URL = os.getenv("ML_TRAINER_URL", "http://localhost:8501")

app = FastAPI(title="DataPilot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

from backend.routes import upload, clean, query, scraper
app.include_router(upload.router,  prefix="/api/upload")
app.include_router(clean.router,   prefix="/api/clean")
app.include_router(query.router,   prefix="/api/query")
app.include_router(scraper.router, prefix="/api/scrape")

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "ml_trainer_url": ML_TRAINER_URL})

@app.get("/scraper")
def scraper_page(request: Request):
    return templates.TemplateResponse("scraper.html", {"request": request, "ml_trainer_url": ML_TRAINER_URL})

@app.get("/analyze")
def analyze_page(request: Request):
    return templates.TemplateResponse("analyze.html", {"request": request, "ml_trainer_url": ML_TRAINER_URL})