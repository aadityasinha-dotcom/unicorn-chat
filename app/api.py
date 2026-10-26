"""FastAPI surface.

POST /chat                    one conversational turn
GET  /sessions/{id}           transcript of a session
DELETE /sessions/{id}         forget a session
GET  /metrics                 in-process counters + latency
GET  /health                  liveness + dataset size
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.bootstrap import build_engine
from app.chatbot.engine import ChatEngine


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message", examples=["What does Razorpay do?"])
    session_id: str | None = Field(None, description="Omit to start a new conversation")


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    kind: str = Field(description="answer | clarification | rejected | error")
    latency_ms: float
    tool_calls: list[str]
    usage: dict[str, int]
    trace_id: str | None = None


def create_app(engine: ChatEngine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or build_engine()
        yield
        app.state.engine.tracer.shutdown()

    app = FastAPI(title="Unicorn Startups Chatbot", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        eng: ChatEngine = app.state.engine
        return {"status": "ok", "companies": len(eng.repo), "model": eng.model.model_name, "langfuse": eng.tracer.enabled}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        eng: ChatEngine = app.state.engine
        result = await run_in_threadpool(eng.chat, req.message, req.session_id)
        return ChatResponse(**{k: getattr(result, k) for k in ChatResponse.model_fields})

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str):
        session = app.state.engine.sessions.get(session_id)
        if not session:
            raise HTTPException(404, "session not found")
        return {"session_id": session.id, "created_at": session.created_at, "turns": [vars(t) for t in session.turns]}

    @app.delete("/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str):
        if not app.state.engine.sessions.delete(session_id):
            raise HTTPException(404, "session not found")

    @app.get("/metrics")
    def get_metrics():
        eng: ChatEngine = app.state.engine
        return {**eng.metrics.snapshot(), "active_sessions": len(eng.sessions)}

    return app


app = create_app()
