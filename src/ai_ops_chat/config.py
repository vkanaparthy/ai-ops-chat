from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_watch_dir: Path = Field(default=Path("./logs_inbox"))
    log_watch_recursive: bool = True
    watch_debounce_seconds: float = 1.5

    chroma_persist_dir: Path = Field(default=Path("./data/chroma"))
    chroma_collection: str = "rca_logs"

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_embed_model: str = "nomic-embed-text"
    # Ollama can block on first embed while loading the model; allow long reads.
    ollama_embed_timeout_seconds: float = 600.0
    # Embed this many texts per /api/embed call. Use 1 if large batches hit ReadTimeout;
    # slightly larger (e.g. 8) is faster when the model is warm.
    ollama_embed_batch_size: int = 8

    aws_region: str = "us-west-2"
    bedrock_model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"

    agent_name: str = "OpsRCA"

    search_default_top_k: int = 10
    list_logs_max_ids: int = 10_000


def get_settings() -> Settings:
    return Settings()
