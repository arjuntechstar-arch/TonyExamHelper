import pytest

from app.models import DocumentChunkDocument, QuestionTemplateDocument
from app.services.generation import GenerationError, GenerationService


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