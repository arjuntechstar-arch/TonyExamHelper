from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_returns_contract_and_request_id() -> None:
    response = client.get("/api/health", headers={"X-Request-ID": "phase-1-test"})

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "api"
    assert response.headers["X-Request-ID"] == "phase-1-test"


def test_invalid_request_id_is_replaced() -> None:
    response = client.get("/api/health", headers={"X-Request-ID": "invalid request id"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != "invalid request id"


def test_unknown_api_path_uses_the_standard_error_contract() -> None:
    response = client.get("/api/not-found", headers={"X-Request-ID": "missing-route"})

    assert response.status_code == 404
    assert response.json() == {
        "error": "http_error",
        "message": "Not Found",
        "request_id": "missing-route",
        "details": None,
    }
