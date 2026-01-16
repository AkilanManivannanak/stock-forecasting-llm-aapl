from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_root_ok():
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "endpoints" in body
