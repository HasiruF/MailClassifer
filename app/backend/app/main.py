from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, corrections, categories, personalization
from app.core.config import settings

app = FastAPI(title="Email Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(corrections.router, prefix="/api", tags=["corrections"])
app.include_router(categories.router, prefix="/api", tags=["categories"])
app.include_router(personalization.router, prefix="/api", tags=["personalization"])
