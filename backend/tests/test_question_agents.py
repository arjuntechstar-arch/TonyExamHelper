import pytest

from app.models import DocumentChunkDocument, QuestionTemplateDocument
from app.services.generation import GenerationError, ProviderRateLimitError
from app.services.question_agents import QuestionGenerationGraph, select_context_window


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


def test_context_window_is_bounded_and_rotates_for_large_papers() -> None:
    paper_chunks = chunks() * 10

    first = select_context_window(paper_chunks, candidate_index=0, max_chunks=3)
    second = select_context_window(paper_chunks, candidate_index=1, max_chunks=3)

    assert len(first) == 3
    assert len(second) == 3
    assert first != second


def test_graph_stops_after_one_rate_limited_provider_request() -> None:
    class RateLimitedProvider:
        provider_name = "rate-limited"

        def __init__(self) -> None:
            self.calls = 0

        def generate_structured(self, prompt: str) -> dict:
            self.calls += 1
            raise ProviderRateLimitError("HTTP 429")

    provider = RateLimitedProvider()
    with pytest.raises(ProviderRateLimitError):
        QuestionGenerationGraph(provider=provider).generate(
            template=template(),
            chunks=chunks(),
            difficulty="Medium",
            bloom_level="Apply",
            candidate_count=5,
        )

    assert provider.calls == 1


@pytest.mark.parametrize(
    ("question_type", "pattern", "difficulty", "bloom_level"),
    [
        ("MCQ", "Direct Concept", "Easy", "Remember"),
        ("MCQ", "Scenario Based", "Medium", "Understand"),
        ("MCQ", "Statement Based", "Hard", "Apply"),
        ("MCQ", "Assertion and Reason", "Easy", "Analyze"),
        ("MCQ", "Case Based", "Medium", "Evaluate"),
        ("MCQ", "Application Based", "Hard", "Create"),
        ("Short Answer", "Define", "Easy", "Remember"),
        ("Short Answer", "Compare", "Medium", "Analyze"),
        ("Long Answer", "Case Study", "Hard", "Evaluate"),
        ("Long Answer", "Problem Solving", "Medium", "Create"),
    ],
)
def test_graph_generates_ten_pattern_difficulty_bloom_combinations(
    question_type: str,
    pattern: str,
    difficulty: str,
    bloom_level: str,
) -> None:
    configured_template = QuestionTemplateDocument(
        name=pattern,
        question_type=question_type,
        pattern=pattern,
        required_fields=["question_text", "explanation"],
        supported_difficulties=[difficulty],
        supported_bloom_levels=[bloom_level],
        version="1.0",
    )

    generated = QuestionGenerationGraph().generate(
        template=configured_template,
        chunks=chunks(),
        difficulty=difficulty,
        bloom_level=bloom_level,
    )

    assert len(generated) == 1
    assert generated[0].question_type == question_type
    assert generated[0].pattern == pattern
    assert generated[0].difficulty == difficulty
    assert generated[0].bloom_level == bloom_level
