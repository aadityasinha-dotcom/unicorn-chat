"""In-memory conversation store.

Each session keeps the *condensed* transcript (user text, assistant text) that
is replayed to the model on later turns. Intermediate tool calls from earlier
turns are not replayed - they are recorded in logs / Langfuse instead. This
keeps the prompt small and makes the cached prefix (system prompt + catalogue)
the dominant cost.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    kind: str = "message"  # message | clarification | rejected | error
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds"))


@dataclass
class Session:
    id: str
    turns: list[Turn] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    pending_clarification: str | None = None  # last question the bot asked, if unanswered

    def history_messages(self, max_turns: int) -> list[dict]:
        """Return the last ``max_turns`` user/assistant pairs as API messages."""
        msgs = [{"role": t.role, "content": t.content} for t in self.turns if t.content]
        msgs = msgs[-(max_turns * 2):]
        # The API requires alternating roles beginning with "user".
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        return msgs


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str | None) -> Session:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            sid = session_id or uuid.uuid4().hex[:12]
            session = Session(id=sid)
            self._sessions[sid] = session
            return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def __len__(self) -> int:
        return len(self._sessions)
