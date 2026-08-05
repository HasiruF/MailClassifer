"""
Central settings, loaded from environment / .env. This backend only ever
receives extracted feature vectors + labels from the client for the opt-in
personalization feature — never raw email subject/body. Keep it that way
when adding new endpoints; see PROJECT_STATUS.md for the privacy design.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/email_classifier"
    cors_origins: list[str] = ["http://localhost:5173"]

    # Threshold at which a user's custom-category corrections graduate from
    # nearest-centroid/kNN scoring to a proper retrained classifier head —
    # see app/services/personalization.py.
    retrain_threshold_per_category: int = 50


settings = Settings()
