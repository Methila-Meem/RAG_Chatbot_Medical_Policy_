from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # API Keys
    groq_api_key: str
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    
    # RAG Configuration
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 3
    cache_expiry: int = 3600
    
    # Model Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "llama-3.1-8b-instant"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()