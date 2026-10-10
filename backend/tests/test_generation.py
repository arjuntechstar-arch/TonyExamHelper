import mongomock
import pytest
from fastapi import HTTPException
from threading import Event
from types import SimpleNamespace

from app.api.questions import GenerateRequest, PaperGenerateRequest, _configured_provider, generate_paper, generate_questions, start_single_question_generation
from app.core.config import Settings
from app.models import DocumentChunkDocument, QuestionTemplateDocument
from app.services.generation import (
    DeterministicLLMProvider,
    FailoverProvider,
    GenerationError,
    GenerationService,
    NvidiaProvider,
    OllamaProvider,
    ProviderRateLimitError,
    parse_chat_completion,
)
from app.services.question_agents import QuestionGenerationGraph
from app.services.generation_runs import generation_runs


def template() -> QuestionTemplateDocument:
    return QuestionTemplateDocument(
        name="Direct Concept",
        question_type="MCQ",
        pattern="Direct Concept",
        required_fields=["question_text", "options", "correct_answer"],
        supported_difficulties=["Medium"],
        supported_bloom_levels=["Apply"],
        version="1.0",
    )


def chunk() -> dict:
    return {
        "chunk": DocumentChunkDocument(
            study_material_id="material-1",
            chunk_index=0,
            page_number=4,
            content="A binary search tree keeps smaller values on the left.",
            metadata={"subject_id": "subject-1"},
        ),
        "score": 0.9,
    }


def test_ollama_provider_calls_openai_compatible_endpoint_and_parses_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [{
                    "message": {
                        "content": '{"question_text":"Example","options":[],"sources":[]}',
                    },
                }],
            }

    def fake_post(url: str, **kwargs) -> Response:
        captured["url"] = url
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr("app.services.generation.httpx.post", fake_post)
    provider = OllamaProvider(
        "https://example-123.ngrok-free.app/",
        model="qwen2.5:32b",
        timeout_seconds=420,
    )

    result = provider.generate_structured("Return a JSON question.")

    assert provider.provider_name == "ollama"
    assert captured["url"] == "https://example-123.ngrok-free.app/v1/chat/completions"
    assert captured["headers"]["ngrok-skip-browser-warning"] == "true"
    assert captured["json"]["model"] == "qwen2.5:32b"
    assert captured["json"]["stream"] is False
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert captured["timeout"] == 420
    assert result["question_text"] == "Example"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://example.ngrok-free.app",
        "https://user:password@example.ngrok-free.app",
        "https://example.ngrok-free.app?token=secret",
    ],
)
def test_ollama_provider_rejects_insecure_or_ambiguous_urls(base_url: str) -> None:
    with pytest.raises(ValueError):
        OllamaProvider(base_url)


def test_ollama_provider_accepts_local_http_and_v1_base_url() -> None:
    provider = OllamaProvider("http://localhost:11434/v1")

    assert provider.endpoint == "http://localhost:11434/v1/chat/completions"


def test_ollama_selection_is_direct_and_does_not_fall_back_to_openrouter() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="ollama",
        ollama_base_url="https://example-123.ngrok-free.app",
        ollama_model="qwen2.5:32b",
        openrouter_api_key="test-openrouter-key",
    )

    provider = _configured_provider(settings)

    assert isinstance(provider, OllamaProvider)
    assert provider.model == "qwen2.5:32b"


def test_ollama_selection_is_unavailable_without_tunnel_url_even_if_openrouter_is_configured() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="ollama",
        openrouter_api_key="test-openrouter-key",
    )

    assert _configured_provider(settings) is None


def test_failover_provider_uses_next_route_only_after_rate_limit() -> None:
    class RateLimitedProvider:
        provider_name = "openrouter"

        def __init__(self) -> None:
            self.calls = 0

        def generate_structured(self, prompt: str) -> dict:
            self.calls += 1
            raise ProviderRateLimitError("HTTP 429")

    class WorkingProvider:
        provider_name = "nvidia-nim"

        def generate_structured(self, prompt: str) -> dict:
            return {"route": "nvidia"}

    limited = RateLimitedProvider()
    provider = FailoverProvider([limited, WorkingProvider()])

    assert provider.generate_structured("question") == {"route": "nvidia"}
    # The first HTTP 429 starts a shared cooldown; the next provider call must
    # skip the exhausted OpenRouter route instead of retrying it again.
    assert provider.generate_structured("question") == {"route": "nvidia"}
    assert limited.calls == 1
    assert provider.active_provider_name == "nvidia-nim"


def test_failover_provider_reports_the_successful_route() -> None:
    class WorkingProvider:
        provider_name = "nvidia-nim"

        def generate_structured(self, prompt: str) -> dict:
            return {"route": "nvidia"}

    routes: list[str] = []
    provider = FailoverProvider([WorkingProvider()], on_route=routes.append)

    assert provider.generate_structured("question") == {"route": "nvidia"}
    assert routes == ["NVIDIA fallback (nvidia-nim)"]


class RetryingProvider:
    provider_name = "test-provider"

    def __init__(self) -> None:
        self.calls = 0

    def generate_structured(self, prompt: str) -> dict:
        self.calls += 1
        if self.calls == 1:
            return {"invalid": True}
        return {
            "question_text": "Where are smaller values placed in a binary search tree?",
            "options": [{"key": "A", "text": "Left subtree"}],
            "correct_answer": "A",
            "explanation": "The source states that smaller values are placed on the left.",
            "difficulty": "Medium",
            "bloom_level": "Apply",
            "sources": [{"chunk_id": "source-1", "page": 4}],
        }


class DuplicateProvider:
    provider_name = "duplicate-provider"

    def generate_structured(self, prompt: str) -> dict:
        return RetryingProvider().generate_structured(prompt)


def test_generation_retries_invalid_structured_output_and_returns_sources() -> None:
    provider = RetryingProvider()
    results = GenerationService(provider, max_retries=1).generate(
        template=template(), chunks=[chunk()], difficulty="Medium", bloom_level="Apply"
    )

    assert provider.calls == 2
    assert results[0].sources[0].page == 4


def test_generation_rejects_unsupported_template_configuration() -> None:
    with pytest.raises(GenerationError, match="difficulty"):
        GenerationService().generate(
            template=template(), chunks=[chunk()], difficulty="Hard", bloom_level="Apply"
        )


def test_generation_suppresses_duplicate_candidates() -> None:
    results = GenerationService(RetryingProvider()).generate(
        template=template(), chunks=[chunk()], difficulty="Medium", bloom_level="Apply", candidate_count=3
    )

    assert len(results) == 1


def test_deterministic_provider_preserves_pipe_characters_in_source() -> None:
    pipe_chunk = chunk()
    pipe_chunk["chunk"].content = "GIS uses layers | maps | and spatial analysis."

    results = GenerationService(DeterministicLLMProvider()).generate(
        template=template(), chunks=[pipe_chunk], difficulty="Medium", bloom_level="Apply"
    )

    assert results[0].sources[0].page == 4
    assert "GIS uses layers" in results[0].options[0].text


def test_question_generation_requires_indexed_source_chunks() -> None:
    database = mongomock.MongoClient().test
    configured_template = template()
    database.question_templates.insert_one(configured_template.model_dump(by_alias=True))

    with pytest.raises(HTTPException, match="No indexed source chunks"):
        generate_questions(
            GenerateRequest(
                template_id=configured_template.id,
                query="binary trees",
                difficulty="Medium",
                bloom_level="Apply",
            ),
            database,
        )


def test_nvidia_provider_parses_non_streaming_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"question_text":"What is a tree?","options":[{"key":"A","text":"A data structure"}],"correct_answer":"A","explanation":"The source defines it.","difficulty":"Medium","bloom_level":"Apply","sources":[{"chunk_id":"chunk-1","page":1}]}'
                        }
                    }
                ]
            }

    def fake_post(*args: object, **kwargs: object) -> Response:
        assert args[0] == "https://integrate.api.nvidia.com/v1/chat/completions"
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["json"]["stream"] is False
        return Response()

    monkeypatch.setattr("app.services.generation.httpx.post", fake_post)
    result = NvidiaProvider("test-key").generate_structured("prompt")

    assert result["question_text"] == "What is a tree?"


def test_nvidia_provider_does_not_duplicate_bearer_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "{}"}}]}

    def fake_post(*args: object, **kwargs: object) -> Response:
        assert kwargs["headers"]["Authorization"] == "Bearer configured-key"
        return Response()

    monkeypatch.setattr("app.services.generation.httpx.post", fake_post)
    assert NvidiaProvider("Bearer configured-key").generate_structured("prompt") == {}


def test_provider_parser_accepts_fenced_json() -> None:
    assert parse_chat_completion({"choices": [{"message": {"content": "```json\n{\"answer\": 42}\n```"}}]}) == {"answer": 42}


def test_paper_generation_falls_back_when_configured_provider_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingProvider:
        provider_name = "failing-provider"

        def generate_structured(self, prompt: str) -> dict:
            raise ValueError("simulated hosted-model failure")

    database = mongomock.MongoClient().test
    configured_template = template()
    configured_template.sections = [{"question_type": "MCQ", "pattern": "Direct Concept", "count": 5, "marks": 1}]
    database.question_templates.insert_one(configured_template.model_dump(by_alias=True))
    for index in range(40):
        document = DocumentChunkDocument(
            study_material_id="material-1",
            chunk_index=index,
            page_number=index + 1,
            content=f"Topic{index} uses mechanism{index} to produce outcome{index}.",
            embedding=[1.0],
        )
        database.document_chunks.insert_one(document.model_dump(by_alias=True))

    monkeypatch.setattr("app.api.questions._configured_provider", lambda settings: FailingProvider())
    graph_generate = QuestionGenerationGraph.generate
    paper_review_modes: list[bool] = []

    def track_paper_reviews(self, *args, **kwargs):
        paper_review_modes.append(kwargs.get("run_llm_review", True))
        return graph_generate(self, *args, **kwargs)

    monkeypatch.setattr("app.services.question_agents.QuestionGenerationGraph.generate", track_paper_reviews)
    trace: list[str] = []
    result = generate_paper(
        PaperGenerateRequest(template_id=configured_template.id, difficulty="Medium", bloom_level="Apply"),
        database,
        object(),
        trace=lambda message, **_: trace.append(message),
    )

    assert len(result) == 5
    assert all(question.sources for question in result)
    assert any("local grounded fallback" in message for message in trace)
    assert paper_review_modes and all(paper_review_modes)


def test_single_question_validation_mode_defaults_to_strict() -> None:
    request = GenerateRequest(
        template_id="template-1",
        query="binary search trees",
        difficulty="Medium",
        bloom_level="Apply",
    )
    assert request.validation_mode == "strict"


@pytest.mark.parametrize(
    ("validation_mode", "expected_attempts", "expected_review"),
    [("fast", 2, False), ("strict", 5, True)],
)
def test_single_question_mode_uses_bounded_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
    validation_mode: str,
    expected_attempts: int,
    expected_review: bool,
) -> None:
    database = mongomock.MongoClient().test
    configured_template = template()
    database.question_templates.insert_one(configured_template.model_dump(by_alias=True))

    class Retrieval:
        def retrieve(self, *_args, **_kwargs):
            return [chunk()]

    observed: dict = {}

    class CapturingGraph:
        def __init__(self, **kwargs) -> None:
            observed.update(kwargs)

        def generate(self, **kwargs):
            observed["run_llm_review"] = kwargs["run_llm_review"]
            return []

    monkeypatch.setattr("app.api.questions._retrieval_service", lambda _database: Retrieval())
    monkeypatch.setattr("app.api.questions._configured_provider", lambda *_args: object())
    monkeypatch.setattr("app.api.questions._existing_questions", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("app.api.questions.QuestionGenerationGraph", CapturingGraph)

    result = generate_questions(
        GenerateRequest(
            template_id=configured_template.id,
            query="binary search trees",
            difficulty="Medium",
            bloom_level="Apply",
            validation_mode=validation_mode,
        ),
        database,
    )

    assert result == []
    assert observed["max_attempts_per_question"] == expected_attempts
    assert observed["max_retries"] == 1
    assert observed["run_llm_review"] is expected_review


def test_single_question_start_returns_live_run_and_persists_worker_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    worker_started = Event()
    allow_worker_to_finish = Event()

    def delayed_generation(payload, database, user_id, trace):
        worker_started.set()
        trace("Retrieving evidence.", stage="retrieval")
        assert allow_worker_to_finish.wait(timeout=2)
        return []

    monkeypatch.setattr("app.api.questions.generate_questions", delayed_generation)
    database = mongomock.MongoClient().test
    user = SimpleNamespace(id="observability-test-user")
    response = start_single_question_generation(
        GenerateRequest(
            template_id="template-1",
            query="binary search trees",
            difficulty="Medium",
            bloom_level="Apply",
        ),
        database,
        user,
    )
    try:
        assert response["request_type"] == "single"
        assert response["status"] in {"queued", "running"}
        assert worker_started.wait(timeout=2)
        run = generation_runs.get(response["id"])
        assert run is not None
        assert run.snapshot()["stage"] == "retrieval"
    finally:
        allow_worker_to_finish.set()

    for _ in range(100):
        run = generation_runs.get(response["id"])
        if run is not None and run.snapshot()["status"] == "completed":
            break
        Event().wait(0.01)
    assert run is not None
    assert run.snapshot()["status"] == "completed"
    assert database.generation_runs.find_one({"_id": response["id"]})["stage"] == "completed"


def test_generation_accepts_legacy_lowercase_difficulty_and_bloom_values() -> None:
    legacy_template = template()
    legacy_template.supported_difficulties = ["medium"]
    legacy_template.supported_bloom_levels = ["apply"]

    results = GenerationService().generate(
        template=legacy_template, chunks=[chunk()], difficulty="Medium", bloom_level="Apply"
    )

    assert results[0].difficulty == "Medium"
    assert results[0].bloom_level == "Apply"


def test_question_generation_reports_unavailable_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.api.questions._configured_provider", lambda settings: None)

    def mock_web_search(query: str, settings: Settings) -> list[dict]:
        assert query == "Distributed Consensus Paxos and Raft"
        assert settings is not None
        return [chunk()]

    monkeypatch.setattr("app.api.questions._web_search_chunks", mock_web_search)
    database = mongomock.MongoClient().test
    configured_template = template()
    database.question_templates.insert_one(configured_template.model_dump(by_alias=True))

    with pytest.raises(HTTPException) as error:
        generate_questions(
            GenerateRequest(
                template_id=configured_template.id,
                query="Distributed Consensus Paxos and Raft",
                difficulty="Medium",
                bloom_level="Apply",
                allow_web_knowledge=True,
            ),
            database,
        )

    assert error.value.status_code == 503
    assert "No AI generation model is available" in str(error.value.detail)
