from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "supply-chain-platform"
    environment: str = "development"

    secret_key: str = "change-this-to-a-long-random-string"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    database_url: str = "postgresql://postgres:postgres@localhost:5432/supplychain"
    redis_url: str = "redis://localhost:6379/0"

    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    model_dir: str = "saved_models"
    forecast_window: int = 30
    forecast_horizon: int = 7

    # NL summary provider: "gemini", "groq", or "" (disabled -> plain-text fallback)
    llm_provider: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )


settings = Settings()