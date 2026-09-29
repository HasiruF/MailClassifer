from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_user: str
    postgres_password: str
    postgres_db: str
    database_port: int = 5433

    google_client_id: str
    google_client_secret: str
    google_redirect_uri: str
    frontend_url: str
    training_data_dir: str = "training_data"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@localhost:{self.database_port}/{self.postgres_db}"
        )


settings = Settings()
