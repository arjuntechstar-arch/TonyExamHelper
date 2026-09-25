import mongomock
import pytest
from hashlib import sha256
from fastapi import HTTPException

from app.api.questions import GenerateRequest, PaperGenerateRequest, generate_paper, generate_questions
from app.models import DocumentChunkDocument, QuestionTemplateDocument
from app.services.generation import (
    DeterministicLLMProvider,
    GenerationError,
    GenerationService,
    NvidiaProvider,
    parse_chat_completion,
)


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
    assert "GIS uses layers" in results[0].question_text


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
            content=" ".join(sha256(f"topic-{index}-{term}".encode()).hexdigest()[:8] for term in range(8)),
            embedding=[1.0],
        )
        database.document_chunks.insert_one(document.model_dump(by_alias=True))

    monkeypatch.setattr("app.api.questions._configured_provider", lambda settings: FailingProvider())
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


def test_generation_accepts_legacy_lowercase_difficulty_and_bloom_values() -> None:
    legacy_template = template()
    legacy_template.supported_difficulties = ["medium"]
    legacy_template.supported_bloom_levels = ["apply"]

    results = GenerationService().generate(
        template=legacy_template, chunks=[chunk()], difficulty="Medium", bloom_level="Apply"
    )

    assert results[0].difficulty == "Medium"
    assert results[0].bloom_level == "Apply"
