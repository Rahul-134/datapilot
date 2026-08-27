import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

ML_TRAINER_URL = os.getenv("ML_TRAINER_URL", "http://localhost:8501")

# Resolve relative to this file, not the process's working directory — Vercel's
# Python runtime does not guarantee cwd is the project root at request time.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "frontend", "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "frontend", "templates")

app = FastAPI(title="DataPilot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

from backend.routes import upload, clean, query, scraper
app.include_router(upload.router,  prefix="/api/upload")
app.include_router(clean.router,   prefix="/api/clean")
app.include_router(query.router,   prefix="/api/query")
app.include_router(scraper.router, prefix="/api/scrape")

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse(request, "index.html", {"ml_trainer_url": ML_TRAINER_URL})

@app.get("/scraper")
def scraper_page(request: Request):
    return templates.TemplateResponse(request, "scraper.html", {"ml_trainer_url": ML_TRAINER_URL})

@app.get("/analyze")
def analyze_page(request: Request):
    return templates.TemplateResponse(request, "analyze.html", {"ml_trainer_url": ML_TRAINER_URL})