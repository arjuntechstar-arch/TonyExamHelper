import pytest

from app.models import DocumentChunkDocument, QuestionTemplateDocument
from app.services.generation import GeneratedQuestion, GenerationError, ProviderRateLimitError, QuestionOption, QuestionSource
from app.services.question_agents import CriticVerdict, LLMQuestionCriticAgent, QuestionGenerationGraph, select_context_window


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


def test_graph_uses_a_new_assessment_angle_when_material_is_reused() -> None:
    first = QuestionGenerationGraph().generate(
        template=template(), chunks=chunks(), difficulty="Medium", bloom_level="Apply"
    )
    second = QuestionGenerationGraph().generate(
        template=template(),
        chunks=chunks(),
        difficulty="Medium",
        bloom_level="Apply",
        existing_questions=first,
    )

    assert len(second) == 1
    assert second[0].question_text != first[0].question_text


def test_graph_balances_mcq_answer_positions() -> None:
    diverse_chunks = [
        {
            "chunk": DocumentChunkDocument(
                study_material_id="material-1", chunk_index=index, page_number=index + 1, content=content,
            ),
            "score": 0.9,
        }
        for index, content in enumerate([
            "Trees place smaller values on left.",
            "Hashes map keys to stored values.",
            "Stacks remove latest items first.",
            "Queues remove earliest items first.",
        ])
    ]
    results = QuestionGenerationGraph().generate(
        template=template(), chunks=diverse_chunks, difficulty="Medium", bloom_level="Apply", candidate_count=4
    )

    assert {question.correct_answer for question in results} == {"A", "B", "C", "D"}
    assert all(len(question.options) == 4 for question in results)


def test_hosted_provider_runs_critic_and_independent_solver_before_acceptance() -> None:
    class VerifiedProvider:
        provider_name = "verified-provider"

        def generate_structured(self, prompt: str) -> dict:
            if "ROLE|QUESTION_CRITIC" in prompt:
                return {
                    "grounded": True, "answerable": True, "single_correct_answer": True,
                    "question_complete": True, "distractors_plausible": True, "contains_source_noise": False,
                    "bloom_match": True, "difficulty_match": True, "mark_match": True,
                    "quality_score": 0.92, "problems": [],
                }
            if "ROLE|INDEPENDENT_ANSWER_SOLVER" in prompt:
                return {"answer": "A", "grounded": True, "rationale": "Evidence supports A."}
            return {
                "question_text": "Which subtree stores smaller values in a binary search tree?",
                "options": [
                    {"key": "A", "text": "Left subtree"}, {"key": "B", "text": "Right subtree"},
                    {"key": "C", "text": "Both subtrees"}, {"key": "D", "text": "Neither subtree"},
                ],
                "correct_answer": "A", "explanation": "Smaller values are stored in the left subtree.",
                "difficulty": "Medium", "bloom_level": "Apply", "sources": [{"chunk_id": "chunk-1", "page": 1}],
            }

    generated = QuestionGenerationGraph(provider=VerifiedProvider()).generate(
        template=template(), chunks=chunks(), difficulty="Medium", bloom_level="Apply"
    )
    assert len(generated) == 1


def test_critic_normalizes_a_ten_point_quality_score() -> None:
    verdict = CriticVerdict.model_validate({
        "grounded": True, "answerable": True, "single_correct_answer": True,
        "question_complete": True, "distractors_plausible": True, "contains_source_noise": False,
        "bloom_match": True, "difficulty_match": True, "mark_match": True,
        "quality_score": 9.5,
    })

    assert verdict.quality_score == 0.95


def test_hosted_critic_rejects_a_semantic_duplicate_across_question_formats() -> None:
    class SemanticDuplicateProvider:
        provider_name = "semantic-duplicate-provider"

        def generate_structured(self, prompt: str) -> dict:
            assert "EXISTING_QUESTION|1|Short Answer" in prompt
            return {
                "grounded": True, "answerable": True, "single_correct_answer": True,
                "question_complete": True, "distractors_plausible": True, "contains_source_noise": False,
                "bloom_match": True, "difficulty_match": True, "mark_match": True,
                "semantic_duplicate": True, "quality_score": 0.95,
                "problems": ["Candidate assesses the same N-gram limitation."],
            }

    candidate = GeneratedQuestion(
        question_text="Why do N-gram models assign low probability to unseen synonym combinations?",
        options=[
            QuestionOption(key="A", text="They rely on observed local sequences."),
            QuestionOption(key="B", text="They use every synonym interchangeably."),
            QuestionOption(key="C", text="They ignore token order."),
            QuestionOption(key="D", text="They only measure punctuation."),
        ],
        correct_answer="A",
        explanation="N-grams estimate from observed local sequences.",
        difficulty="Medium",
        bloom_level="Understand",
        sources=[QuestionSource(chunk_id="chunk-1", page=1)],
    )
    existing = candidate.model_copy(update={
        "question_type": "Short Answer",
        "options": [],
        "correct_answer": None,
        "expected_answer": "N-grams rely on observed sequences and therefore struggle with unseen synonym combinations.",
    })

    issues = LLMQuestionCriticAgent(SemanticDuplicateProvider()).validate(
        candidate,
        context=["N-gram models estimate from observed local token sequences."],
        template=template(),
        existing_questions=[existing],
    )

    assert len(issues) == 1
    assert issues[0].code == "llm_critic"
    assert "same N-gram limitation" in issues[0].message


def test_solver_disagreement_rejects_a_hosted_candidate() -> None:
    class DisagreeingProvider:
        provider_name = "disagreeing-provider"

        def generate_structured(self, prompt: str) -> dict:
            if "ROLE|QUESTION_CRITIC" in prompt:
                return {
                    "grounded": True, "answerable": True, "single_correct_answer": True,
                    "question_complete": True, "distractors_plausible": True, "contains_source_noise": False,
                    "bloom_match": True, "difficulty_match": True, "mark_match": True,
                    "quality_score": 0.95,
                }
            if "ROLE|INDEPENDENT_ANSWER_SOLVER" in prompt:
                return {"answer": "B", "grounded": True}
            return {
                "question_text": "Which subtree stores smaller values in a binary search tree?",
                "options": [
                    {"key": "A", "text": "Left subtree"}, {"key": "B", "text": "Right subtree"},
                    {"key": "C", "text": "Both subtrees"}, {"key": "D", "text": "Neither subtree"},
                ],
                "correct_answer": "A", "explanation": "Smaller values are stored in the left subtree.",
                "difficulty": "Medium", "bloom_level": "Apply", "sources": [{"chunk_id": "chunk-1", "page": 1}],
            }

    # The provider repeats the same candidate; every one is rejected by the solver.
    with pytest.raises(GenerationError, match="solver_disagreement"):
        QuestionGenerationGraph(provider=DisagreeingProvider(), max_attempts_per_question=1).generate(
            template=template(), chunks=chunks(), difficulty="Medium", bloom_level="Apply"
        )


def test_graph_returns_valid_partial_paper_when_one_candidate_exhausts_retries() -> None:
    class RepeatingProvider:
        provider_name = "repeating-provider"

        def generate_structured(self, prompt: str) -> dict:
            if "ROLE|QUESTION_CRITIC" in prompt:
                return {"grounded": True, "answerable": True, "single_correct_answer": True, "question_complete": True, "distractors_plausible": True, "contains_source_noise": False, "bloom_match": True, "difficulty_match": True, "mark_match": True, "quality_score": 0.95}
            if "ROLE|INDEPENDENT_ANSWER_SOLVER" in prompt:
                return {"answer": "A", "grounded": True}
            return {
                "question_text": "Which subtree stores smaller values in a binary search tree?",
                "options": [
                    {"key": "A", "text": "Left subtree"}, {"key": "B", "text": "Right subtree"},
                    {"key": "C", "text": "Both subtrees"}, {"key": "D", "text": "Neither subtree"},
                ],
                "correct_answer": "A", "explanation": "Smaller values are stored in the left subtree.",
                "difficulty": "Medium", "bloom_level": "Apply", "sources": [{"chunk_id": "chunk-1", "page": 1}],
            }

    trace: list[str] = []
    result = QuestionGenerationGraph(provider=RepeatingProvider(), max_attempts_per_question=1).generate(
        template=template(), chunks=chunks(), difficulty="Medium", bloom_level="Apply", candidate_count=2,
        allow_partial=True, trace=lambda message, **_: trace.append(message),
    )

    assert len(result) == 1
    assert any("Returned 1 of 2" in message for message in trace)


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
        marks=1 if question_type == "MCQ" else 2,
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
    if question_type != "MCQ":
        assert generated[0].expected_answer
