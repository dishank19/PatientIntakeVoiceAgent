from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str
    DAILY_API_KEY: str
    CARTESIA_API_KEY: Optional[str] = None
    
    # Daily.co Configuration
    DAILY_ROOM_URL: str
    DAILY_ROOM_NAME: str
    
    # OpenAI Configuration
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_TEMPERATURE: float = 0.7
    
    # Audio Configuration
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    # Data Storage
    DATA_DIR: str = "data"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings() 