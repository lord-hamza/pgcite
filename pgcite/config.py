from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    model_cache_dir: str = ".cache/models"

    generator: Literal["extractive", "anthropic"] = "extractive"
    llm_model: str = "claude-opus-5-5"
    llm_effort: Literal["low", "medium", "high"] = "low"
    llm_max_tokens: int = 2048
    anthropic_api_key: str | None = None

    chunk_words: int = 120
    candidates: int = 40
    rerank_candidates: int = 20
    top_k: int = 5

    api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
