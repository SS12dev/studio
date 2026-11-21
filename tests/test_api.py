from fastapi.testclient import TestClient

from studio.api.main import app


def test_health_and_models():
    c = TestClient(app)
    assert c.get("/health").json()["ok"] is True
    body = c.get("/models").json()
    assert body["budget_gb"] > 0 and body["loaded"] == []
