from fastapi.testclient import TestClient

from app.api.health import health
from app.core.config import Settings
from app.main import app


client = TestClient(app)


def test_health_reports_selected_ollama_model_without_openrouter_fallback() -> None:
    response = health(Settings(
        _env_file=None,
        llm_provider="ollama",
        ollama_base_url="https://example-123.ngrok-free.app",
        ollama_model="qwen2.5:32b",
        openrouter_api_key="test-openrouter-key",
    ))

    assert response.model_status == "configured"
    assert response.model_provider == "Ollama"
    assert response.model_name == "qwen2.5:32b"


def test_health_reports_ollama_unavailable_until_tunnel_url_is_set() -> None:
    response = health(Settings(_env_file=None, llm_provider="ollama"))

    assert response.model_status == "unavailable"
    assert response.model_provider == "Ollama"
    assert response.model_name is None


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
