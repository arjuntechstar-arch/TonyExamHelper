from fastapi.testclient import TestClient

from app.main import app, is_generation_status_poll


client = TestClient(app)


def test_security_headers_are_present_on_api_responses() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


def test_rate_limit_blocks_excessive_requests() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200

    for _ in range(105):
        client.get("/api/health")

    throttled = client.get("/api/health")
    assert throttled.status_code == 429
    assert throttled.json()["error"] == "rate_limit"


def test_generation_status_poll_is_exempt_from_general_rate_limit() -> None:
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/questions/generate/runs/run-123",
        "headers": [],
    }
    assert is_generation_status_poll(Request(scope))
