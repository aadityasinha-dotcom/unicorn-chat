"""Wire the application together (shared by the API and the CLI)."""
from __future__ import annotations

import logging

from dotenv import load_dotenv

from app.chatbot.engine import ChatEngine
from app.chatbot.llm import AnthropicChatModel
from app.config import Settings, get_settings
from app.data.loader import load_companies
from app.data.repository import CompanyRepository
from app.logging_setup import setup_logging
from app.observability.metrics import metrics
from app.observability.tracing import Tracer

log = logging.getLogger(__name__)


def build_engine(settings: Settings | None = None) -> ChatEngine:
    load_dotenv()
    settings = settings or get_settings()
    setup_logging(settings.log_dir, settings.log_level)
    repo = CompanyRepository(load_companies(settings.dataset_path))
    log.info("dataset loaded", extra={"companies": len(repo), "path": str(settings.dataset_path)})
    if not settings.anthropic_api_key:
        log.warning("ANTHROPIC_API_KEY is not set - the SDK will try other credential sources; calls may fail with 401")
    model = AnthropicChatModel(settings)
    tracer = Tracer(settings)
    return ChatEngine(repo=repo, model=model, settings=settings, metrics=metrics, tracer=tracer)
