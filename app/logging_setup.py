"""Structured JSON logging.

Every record is one JSON object per line, written to stdout and to
``<log_dir>/app.jsonl``. Conversation events (``event=user_query`` /
``event=bot_response``) additionally go to ``<log_dir>/conversations.jsonl``
so they can be analysed independently of operational noise.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

CONVERSATION_LOGGER = "unicorn.conversation"
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module", "exc_info",
    "exc_text", "stack_info", "lineno", "funcName", "created", "msecs", "relativeCreated", "thread",
    "threadName", "processName", "process", "message", "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = JsonFormatter()

    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    app_file = logging.FileHandler(log_dir / "app.jsonl", encoding="utf-8")
    app_file.setFormatter(formatter)
    root.addHandler(app_file)

    conv = logging.getLogger(CONVERSATION_LOGGER)
    conv.setLevel(logging.INFO)
    conv_file = logging.FileHandler(log_dir / "conversations.jsonl", encoding="utf-8")
    conv_file.setFormatter(formatter)
    conv.addHandler(conv_file)

    # Quieten chatty third-party loggers.
    for noisy in ("httpx", "httpcore", "anthropic", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
