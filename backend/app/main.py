from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.ml import router as ml_router

app = FastAPI(
    title="AI & ML Workbook API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ml_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "ai-ml-workbook-api"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}