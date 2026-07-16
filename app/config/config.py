import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # App Settings
    app_name: str = Field(default="VoltIQ", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    secret_key: str = Field(default="dev_secret_key_change_in_production_12345", env="SECRET_KEY")
    
    # Database Settings
    database_url: str = Field(default="sqlite:///./voltiq.db", env="DATABASE_URL")
    
    # AI/LLM Settings
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_model_name: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL_NAME")
    openai_model_temp: float = Field(default=0.0, env="OPENAI_MODEL_TEMP")
    
    # Data Reference Configuration
    datasets_dir: str = Field(default="datasets", env="DATASETS_DIR")

    # Load from .env file
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
