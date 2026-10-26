from fastapi.testclient import TestClient

from app.api import create_app
from app.chatbot.engine import ChatEngine
from app.observability.metrics import Metrics
from tests.conftest import FakeModel, text_block, tool_block


def test_chat_roundtrip_and_metrics(repo, settings):
    model = FakeModel([tool_block("get_company", {"name": "Zomato"})], [text_block("Zomato is a food delivery platform.")])
    engine = ChatEngine(repo=repo, model=model, settings=settings, metrics=Metrics())
    with TestClient(create_app(engine)) as client:
        assert client.get("/health").json()["companies"] == 118
        r = client.post("/chat", json={"message": "What does Zomato do?"})
        assert r.status_code == 200
        body = r.json()
        assert body["kind"] == "answer" and body["tool_calls"] == ["get_company"]
        sid = body["session_id"]
        hist = client.get(f"/sessions/{sid}").json()
        assert [t["role"] for t in hist["turns"]] == ["user", "assistant"]
        m = client.get("/metrics").json()
        assert m["counters"]["queries_total"] == 1 and m["latency"]["count"] == 1 and m["active_sessions"] == 1
        assert client.delete(f"/sessions/{sid}").status_code == 204
        assert client.get(f"/sessions/{sid}").status_code == 404
        assert client.post("/chat", json={"message": ""}).json()["kind"] == "rejected"
        assert client.post("/chat", json={}).status_code == 422
