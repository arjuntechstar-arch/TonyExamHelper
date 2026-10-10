import pytest
from fastapi import HTTPException

from app.api.questions import _attach_web_citations, _web_search_chunks
from app.core.config import Settings
from app.models import DocumentChunkDocument
from app.services.generation import GeneratedQuestion
from app.services.web_search import TavilySearchProvider, WebSearchError


def test_tavily_search_returns_bounded_web_chunks_and_ignores_invalid_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {
                        "title": "Reliable source",
                        "url": "https://example.org/topic",
                        "content": "A reliable source explains the topic and its applications.",
                        "score": 0.92,
                    },
                    {
                        "title": "Unsafe source",
                        "url": "javascript:alert(1)",
                        "content": "Must not be used.",
                    },
                ],
            }

    def fake_post(url: str, **kwargs: object) -> Response:
        assert url == TavilySearchProvider.endpoint
        request_body = kwargs["json"]
        assert isinstance(request_body, dict)
        assert request_body["query"] == "topic"
        assert request_body["max_results"] == 3
        assert kwargs["timeout"] == TavilySearchProvider.timeout_seconds
        return Response()

    monkeypatch.setattr("app.services.web_search.httpx.post", fake_post)
    chunks = TavilySearchProvider("test-key").search("topic", max_results=3)

    assert len(chunks) == 1
    assert chunks[0]["chunk"].content == "A reliable source explains the topic and its applications."
    assert chunks[0]["chunk"].metadata["url"] == "https://example.org/topic"
    assert chunks[0]["chunk"].metadata["source"] == "web_search"


def test_web_search_reports_provider_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    def timeout(*args: object, **kwargs: object) -> None:
        assert args == (TavilySearchProvider.endpoint,)
        assert kwargs["timeout"] == TavilySearchProvider.timeout_seconds
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr("app.services.web_search.httpx.post", timeout)

    with pytest.raises(WebSearchError, match="timed out"):
        TavilySearchProvider("test-key").search("topic")


def test_web_search_requires_configuration() -> None:
    with pytest.raises(HTTPException) as error:
        _web_search_chunks("topic", Settings(_env_file=None, tavily_api_key=None))

    assert error.value.status_code == 503
    assert "TAVILY_API_KEY" in error.value.detail


def test_generated_question_carries_search_source_citation() -> None:
    chunk = DocumentChunkDocument(
        id="web-source-1",
        study_material_id="web-search",
        chunk_index=0,
        page_number=1,
        content="A source explains the topic with sufficient detail.",
        metadata={"source": "web_search", "title": "Topic guide", "url": "https://example.org/guide"},
    )
    question = GeneratedQuestion(
        question_text="What does the topic explain?",
        options=[],
        correct_answer=None,
        expected_answer="It explains the topic.",
        explanation="The cited source explains the topic.",
        difficulty="Medium",
        bloom_level="Apply",
        sources=[{"chunk_id": chunk.id, "page": 1}],
    )

    result = _attach_web_citations([question], [{"chunk": chunk, "score": 0.9}])

    assert result[0].sources[0].title == "Topic guide"
    assert result[0].sources[0].url == "https://example.org/guide"
