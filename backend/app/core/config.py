import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import validator

class Settings(BaseSettings):
    # API settings
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Carbon Credit Verification"
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Database settings
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL", "sqlite:///./carbon_credits.db")
    
    # PostgreSQL settings
    POSTGRES_USER: Optional[str] = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: Optional[str] = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: Optional[str] = os.getenv("POSTGRES_DB", "carbon_credits")
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "changethisinproduction")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # Blockchain settings
    BLOCKCHAIN_PROVIDER_URL: str = os.getenv("BLOCKCHAIN_PROVIDER_URL", "https://polygon-mumbai.infura.io/v3/your-infura-key")
    BLOCKCHAIN_PRIVATE_KEY: str = os.getenv("BLOCKCHAIN_PRIVATE_KEY", "")
    CONTRACT_ADDRESS: str = os.getenv("CONTRACT_ADDRESS", "")
    
    # Satellite imagery settings
    SENTINEL_API_URL: str = "https://services.sentinel-hub.com/api/v1/"
    SENTINEL_API_KEY: str = os.getenv("SENTINEL_API_KEY", "")
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # SQLite configuration
    @validator("DATABASE_URL", pre=True)
    def validate_database_url(cls, v: Optional[str]) -> str:
        if v and v.startswith("sqlite"):
            return v
        return v or "sqlite:///./carbon_credits.db"
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
