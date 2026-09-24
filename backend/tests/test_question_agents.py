import pytest

from app.models import DocumentChunkDocument, QuestionTemplateDocument
from app.services.generation import GenerationError
from app.services.question_agents import QuestionGenerationGraph


def template() -> QuestionTemplateDocument:
    return QuestionTemplateDocument(
        name="Direct Concept",
        question_type="MCQ",
        pattern="Direct Concept",
        required_fields=["question_text", "options", "correct_answer", "explanation"],
        supported_difficulties=["Medium"],
        supported_bloom_levels=["Apply"],
        version="1.0",
    )


def chunks() -> list[dict]:
    return [
        {
            "chunk": DocumentChunkDocument(
                study_material_id="material-1",
                chunk_index=index,
                page_number=index + 1,
                content=content,
            ),
            "score": 0.9,
        }
        for index, content in enumerate(
            [
                "A binary search tree keeps smaller values on the left.",
                "A binary search tree keeps larger values on the right.",
            ]
        )
    ]


def test_graph_generates_distinct_grounded_candidates() -> None:
    results = QuestionGenerationGraph().generate(
        template=template(),
        chunks=chunks(),
        difficulty="Medium",
        bloom_level="Apply",
        candidate_count=2,
    )

    assert len(results) == 2
    assert len({item.question_text for item in results}) == 2
    assert all(item.pattern == "Direct Concept" for item in results)
    assert all(item.sources for item in results)


def test_graph_rejects_unsupported_configuration_before_generation() -> None:
    with pytest.raises(GenerationError, match="difficulty"):
        QuestionGenerationGraph().generate(
            template=template(),
            chunks=chunks(),
            difficulty="Hard",
            bloom_level="Apply",
        )
