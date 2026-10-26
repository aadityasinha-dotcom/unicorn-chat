"""Application settings, loaded from environment variables / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: str = "medium"  # low | medium | high
    anthropic_enable_fallbacks: bool = True
    anthropic_max_tokens: int = 4096

    # Observability
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # App
    dataset_path: Path = Path("data/unicorns_india_may2025.csv")
    log_dir: Path = Path("logs")
    log_level: str = "INFO"
    max_input_chars: int = 2000
    max_tool_rounds: int = 6
    max_history_turns: int = 20

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
