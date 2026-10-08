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

    # NL / LLM provider: "gemini", "groq", or "" (disabled -> plain-text fallback)
    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    groq_api_key: str = ""

    # Brevo transactional email configuration
    brevo_api_key: str = ""
    brevo_sender_email: str = ""
    brevo_sender_name: str = "Supply Chain Platform"

    # CORS: comma-separated list of allowed origins for production.
    # In development (environment=="development"), a permissive wildcard is used.
    # Example: CORS_ALLOWED_ORIGINS="https://app.example.com,https://admin.example.com"
    cors_allowed_origins: str = ""

    # Logging level: DEBUG | INFO | WARNING | ERROR
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    def get_cors_origins(self) -> list[str]:
        """
        Returns the list of allowed CORS origins.
        - Development: permissive (localhost variants).
        - Production: only explicitly configured origins.
        """
        if self.cors_allowed_origins:
            return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]
        if self.environment == "development":
            return [
                "http://localhost",
                "http://localhost:3000",
                "http://localhost:5173",
                "http://localhost:8000",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173",
                "http://127.0.0.1:8000",
            ]
        # Production with no origins configured: deny all cross-origin requests.
        return []


settings = Settings()