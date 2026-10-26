from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.chatbot.engine import ChatEngine
from app.chatbot.llm import ModelResponse, Usage
from app.config import Settings
from app.data.loader import load_companies
from app.data.repository import CompanyRepository
from app.observability.metrics import Metrics

DATASET = Path(__file__).resolve().parents[1] / "data" / "unicorns_india_may2025.csv"


@pytest.fixture(scope="session")
def repo() -> CompanyRepository:
    return CompanyRepository(load_companies(DATASET))


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(anthropic_api_key="test", log_dir=tmp_path / "logs", dataset_path=DATASET, _env_file=None)


def text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def tool_block(name: str, inp: dict, tid: str = "tu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=inp, id=tid)


class FakeModel:
    """Scripted model: returns queued responses in order and records every request."""

    model_name = "fake-model"

    def __init__(self, *scripts: list):
        self.queue = [ModelResponse("tool_use" if any(b.type == "tool_use" for b in c) else "end_turn", list(c), Usage(10, 5), "fake-model") for c in scripts]
        self.requests: list[dict] = []

    def create(self, *, system, messages, tools):
        self.requests.append({"system": system, "messages": [dict(m) for m in messages], "tools": tools})
        if not self.queue:
            raise AssertionError("FakeModel ran out of scripted responses")
        return self.queue.pop(0)


@pytest.fixture
def make_engine(repo, settings):
    def _make(*scripts):
        model = FakeModel(*scripts)
        return ChatEngine(repo=repo, model=model, settings=settings, metrics=Metrics()), model

    return _make
