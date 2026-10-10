from threading import Barrier

import pytest

from app.models import DocumentChunkDocument, QuestionTemplateDocument
from app.services.generation import GeneratedQuestion, GenerationError, ProviderRateLimitError, QuestionOption, QuestionSource
from app.services.question_agents import CriticVerdict, LLMQuestionCriticAgent, QuestionGenerationGraph, select_context_window
from app.services.generation_decision import JevDecision, JevDecisionAgent, JevDecisionPolicy
from app.services.quality import ValidationResult


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
    reviewer_barrier = Barrier(2)

    class VerifiedProvider:
        provider_name = "verified-provider"

        def generate_structured(self, prompt: str) -> dict:
            if "ROLE|QUESTION_CRITIC" in prompt:
                reviewer_barrier.wait(timeout=3)
                return {
                    "grounded": True, "answerable": True, "single_correct_answer": True,
                    "question_complete": True, "distractors_plausible": True, "contains_source_noise": False,
                    "bloom_match": True, "difficulty_match": True, "mark_match": True,
                    "quality_score": 0.92, "problems": [],
                }
            if "ROLE|INDEPENDENT_ANSWER_SOLVER" in prompt:
                reviewer_barrier.wait(timeout=3)
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

    trace_events: list[dict] = []
    generated = QuestionGenerationGraph(provider=VerifiedProvider()).generate(
        template=template(),
        chunks=chunks(),
        difficulty="Medium",
        bloom_level="Apply",
        trace=lambda message, **kwargs: trace_events.append({"message": message, **kwargs}),
    )
    assert len(generated) == 1
    assert {event["stage"] for event in trace_events if event.get("duration_ms") is not None} == {
        "generation_model",
        "critic_model",
        "solver_model",
    }
    assert sum(
        event["metric_deltas"]["model_calls"]
        for event in trace_events
        if event.get("metric_deltas") and "model_calls" in event["metric_deltas"]
    ) == 3


def test_fast_validation_skips_hosted_review_but_keeps_local_validation() -> None:
    class GenerationOnlyProvider:
        provider_name = "generation-only"

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate_structured(self, prompt: str) -> dict:
            self.prompts.append(prompt)
            return {
                "question_text": "Which subtree stores smaller values in a binary search tree?",
                "options": [
                    {"key": "A", "text": "Left subtree"}, {"key": "B", "text": "Right subtree"},
                    {"key": "C", "text": "Both subtrees"}, {"key": "D", "text": "Neither subtree"},
                ],
                "correct_answer": "A", "explanation": "Smaller values are stored in the left subtree.",
                "difficulty": "Medium", "bloom_level": "Apply", "sources": [{"chunk_id": "chunk-1", "page": 1}],
            }

    provider = GenerationOnlyProvider()
    trace_events: list[dict] = []
    generated = QuestionGenerationGraph(provider=provider).generate(
        template=template(),
        chunks=chunks(),
        difficulty="Medium",
        bloom_level="Apply",
        run_llm_review=False,
        trace=lambda message, **kwargs: trace_events.append({"message": message, **kwargs}),
    )

    assert len(generated) == 1
    assert len(provider.prompts) == 1
    assert "ROLE|QUESTION_CRITIC" not in provider.prompts[0]
    assert "ROLE|INDEPENDENT_ANSWER_SOLVER" not in provider.prompts[0]
    assert any(event.get("stage") == "question_validation" for event in trace_events)
    assert any(event.get("stage") == "llm_review_skipped" for event in trace_events)
    jev_event = next(event for event in trace_events if event.get("stage") == "jev_decision")
    assert jev_event["metric_deltas"]["jev_review_escalations"] == 0
    assert "evidence confidence" in jev_event["message"]


def test_jev_escalates_fast_mode_for_advanced_assessments() -> None:
    class RiskAwareProvider:
        provider_name = "risk-aware"

        def __init__(self) -> None:
            self.roles: list[str] = []

        def generate_structured(self, prompt: str) -> dict:
            if "ROLE|QUESTION_CRITIC" in prompt:
                self.roles.append("critic")
                return {
                    "grounded": True, "answerable": True, "single_correct_answer": True,
                    "question_complete": True, "distractors_plausible": True, "contains_source_noise": False,
                    "bloom_match": True, "difficulty_match": True, "mark_match": True,
                    "quality_score": 0.92, "problems": [],
                }
            if "ROLE|INDEPENDENT_ANSWER_SOLVER" in prompt:
                self.roles.append("solver")
                return {"answer": "A", "grounded": True, "rationale": "Evidence supports A."}
            self.roles.append("generation")
            return {
                "question_text": "How does a binary search tree determine whether to place a new value in the left or right subtree?",
                "options": [
                    {"key": "A", "text": "Left subtree"}, {"key": "B", "text": "Right subtree"},
                    {"key": "C", "text": "Both subtrees"}, {"key": "D", "text": "Neither subtree"},
                ],
                "correct_answer": "A",
                "explanation": "Smaller values are stored in the left subtree.",
                "difficulty": "Hard",
                "bloom_level": "Analyze",
                "sources": [{"chunk_id": "chunk-1", "page": 1}],
            }

    provider = RiskAwareProvider()
    high_risk_template = template().model_copy(update={
        "marks": 2,
        "supported_difficulties": ["Hard"],
        "supported_bloom_levels": ["Analyze"],
    })
    trace_events: list[dict] = []

    generated = QuestionGenerationGraph(provider=provider).generate(
        template=high_risk_template,
        chunks=chunks(),
        difficulty="Hard",
        bloom_level="Analyze",
        run_llm_review=False,
        trace=lambda message, **kwargs: trace_events.append({"message": message, **kwargs}),
    )

    assert len(generated) == 1
    assert provider.roles == ["generation", "critic", "solver"]
    decision_event = next(event for event in trace_events if event.get("stage") == "jev_decision")
    assert decision_event["metric_deltas"]["jev_review_escalations"] == 1
    assert "high_difficulty" in decision_event["message"]
    assert "advanced_bloom_level" in decision_event["message"]
    assert "multi_mark_question" in decision_event["message"]


def test_jev_decision_is_typed_and_low_evidence_escalates() -> None:
    question = GeneratedQuestion(
        question_text="What is the color of the sky?",
        options=[
            QuestionOption(key="A", text="Blue"), QuestionOption(key="B", text="Green"),
            QuestionOption(key="C", text="Red"), QuestionOption(key="D", text="Yellow"),
        ],
        correct_answer="A",
        explanation="This is a test answer.",
        difficulty="Medium",
        bloom_level="Apply",
        sources=[QuestionSource(chunk_id="unknown", page=9)],
    )
    validation = ValidationResult(
        valid=True,
        score=0.7,
        ranking_score=0.7,
        similarity=0.0,
    )

    decision = JevDecisionAgent().decide(
        question,
        template=template(),
        context_chunks=chunks(),
        validation=validation,
        strict_review_requested=False,
    )

    assert isinstance(decision, JevDecision)
    assert decision.action == "review_with_hosted_checks"
    assert decision.review_required is True
    assert decision.evidence_confidence < JevDecisionPolicy().minimum_evidence_confidence
    assert decision.reasons == ["low_evidence_confidence"]


def test_jev_requests_one_more_retrieval_for_low_evidence_fast_candidate() -> None:
    question = GeneratedQuestion(
        question_text="What is the color of the sky?",
        options=[
            QuestionOption(key="A", text="Blue"), QuestionOption(key="B", text="Green"),
            QuestionOption(key="C", text="Red"), QuestionOption(key="D", text="Yellow"),
        ],
        correct_answer="A",
        explanation="This is a test answer.",
        difficulty="Medium",
        bloom_level="Apply",
        sources=[QuestionSource(chunk_id="unknown", page=9)],
    )
    validation = ValidationResult(valid=True, score=0.7, ranking_score=0.7, similarity=0.0)

    decision = JevDecisionAgent().decide(
        question,
        template=template(),
        context_chunks=chunks(),
        validation=validation,
        strict_review_requested=False,
        allow_retrieval_expansion=True,
    )
    strict_decision = JevDecisionAgent().decide(
        question,
        template=template(),
        context_chunks=chunks(),
        validation=validation,
        strict_review_requested=True,
        allow_retrieval_expansion=True,
    )

    assert decision.action == "retrieve_more_evidence"
    assert decision.review_required is False
    assert strict_decision.action == "review_with_hosted_checks"
    assert strict_decision.review_required is True


def test_graph_retrieves_once_then_retries_with_expanded_context() -> None:
    class ExpansionProvider:
        provider_name = "expansion-provider"

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate_structured(self, prompt: str) -> dict:
            self.prompts.append(prompt)
            return {
                "question_text": "Which subtree stores smaller values in a binary search tree?",
                "options": [
                    {"key": "A", "text": "Left subtree"}, {"key": "B", "text": "Right subtree"},
                    {"key": "C", "text": "Both subtrees"}, {"key": "D", "text": "Neither subtree"},
                ],
                "correct_answer": "A",
                "explanation": "Smaller values are stored in the left subtree.",
                "difficulty": "Medium",
                "bloom_level": "Apply",
                "sources": [{"chunk_id": "chunk-3", "page": 3}],
            }

    class ExpansionDecisionAgent:
        def __init__(self) -> None:
            self.calls = 0

        def decide(self, _question, **_kwargs) -> JevDecision:
            self.calls += 1
            if self.calls == 1:
                return JevDecision(
                    action="retrieve_more_evidence",
                    evidence_confidence=0.4,
                    review_required=False,
                    reasons=["low_evidence_confidence"],
                )
            return JevDecision(
                action="accept_locally",
                evidence_confidence=0.95,
                review_required=False,
                reasons=[],
            )

    provider = ExpansionProvider()
    graph = QuestionGenerationGraph(provider=provider)
    graph.decision_agent = ExpansionDecisionAgent()
    additional_chunk = {
        "chunk": DocumentChunkDocument(
            id="chunk-3",
            study_material_id="material-1",
            chunk_index=3,
            page_number=3,
            content="A binary search tree stores values smaller than a node in its left subtree.",
        ),
        "score": 0.95,
    }
    retrieval_calls: list[str] = []
    trace_events: list[dict] = []

    generated = graph.generate(
        template=template(),
        chunks=chunks(),
        difficulty="Medium",
        bloom_level="Apply",
        run_llm_review=False,
        retrieve_additional_evidence=lambda question: (
            retrieval_calls.append(question.question_text) or [additional_chunk]
        ),
        trace=lambda message, **kwargs: trace_events.append({"message": message, **kwargs}),
    )

    assert len(generated) == 1
    assert len(retrieval_calls) == 1
    assert len(provider.prompts) == 2
    assert "A binary search tree stores values smaller than a node in its left subtree." in provider.prompts[1]
    expansion_event = next(event for event in trace_events if event.get("stage") == "jev_retrieval")
    assert expansion_event["metric_deltas"]["jev_retrieval_expansions"] == 1


def test_jev_policy_accepts_grounded_low_risk_fast_question() -> None:
    source = chunks()[0]["chunk"]
    question = GeneratedQuestion(
        question_text="Which subtree stores smaller values in a binary search tree?",
        options=[
            QuestionOption(key="A", text="Left subtree"), QuestionOption(key="B", text="Right subtree"),
            QuestionOption(key="C", text="Both subtrees"), QuestionOption(key="D", text="Neither subtree"),
        ],
        correct_answer="A",
        explanation="Smaller values are stored in the left subtree.",
        difficulty="Medium",
        bloom_level="Apply",
        sources=[QuestionSource(chunk_id=source.id, page=source.page_number)],
    )
    validation = ValidationResult(valid=True, score=0.8, ranking_score=0.8, similarity=0.4)
    decision = JevDecisionAgent().decide(
        question,
        template=template(),
        context_chunks=chunks(),
        validation=validation,
        strict_review_requested=False,
    )

    assert decision.action == "accept_locally"
    assert decision.review_required is False
    assert decision.evidence_confidence >= JevDecisionPolicy().minimum_evidence_confidence
    assert decision.reasons == []


def test_jev_escalates_questions_grounded_in_external_web_evidence() -> None:
    source = chunks()[0]["chunk"].model_copy(update={
        "metadata": {"source": "web_search"},
    })
    question = GeneratedQuestion(
        question_text="Which subtree stores smaller values in a binary search tree?",
        options=[
            QuestionOption(key="A", text="Left subtree"), QuestionOption(key="B", text="Right subtree"),
            QuestionOption(key="C", text="Both subtrees"), QuestionOption(key="D", text="Neither subtree"),
        ],
        correct_answer="A",
        explanation="Smaller values are stored in the left subtree.",
        difficulty="Medium",
        bloom_level="Apply",
        sources=[QuestionSource(chunk_id=source.id, page=source.page_number)],
    )
    decision = JevDecisionAgent().decide(
        question,
        template=template(),
        context_chunks=[{"chunk": source, "score": 1.0}],
        validation=ValidationResult(valid=True, score=0.8, ranking_score=0.8, similarity=0.4),
        strict_review_requested=False,
    )

    assert decision.evidence_confidence >= JevDecisionPolicy().minimum_evidence_confidence
    assert decision.action == "review_with_hosted_checks"
    assert decision.reasons == ["external_web_evidence"]


def test_jev_always_honors_strict_review_mode() -> None:
    source = chunks()[0]["chunk"]
    question = GeneratedQuestion(
        question_text="Which subtree stores smaller values in a binary search tree?",
        options=[
            QuestionOption(key="A", text="Left subtree"), QuestionOption(key="B", text="Right subtree"),
            QuestionOption(key="C", text="Both subtrees"), QuestionOption(key="D", text="Neither subtree"),
        ],
        correct_answer="A",
        explanation="Smaller values are stored in the left subtree.",
        difficulty="Medium",
        bloom_level="Apply",
        sources=[QuestionSource(chunk_id=source.id, page=source.page_number)],
    )
    decision = JevDecisionAgent().decide(
        question,
        template=template(),
        context_chunks=chunks(),
        validation=ValidationResult(valid=True, score=0.8, ranking_score=0.8, similarity=0.4),
        strict_review_requested=True,
    )

    assert decision.action == "review_with_hosted_checks"
    assert decision.evidence_confidence >= JevDecisionPolicy().minimum_evidence_confidence
    assert decision.reasons == ["strict_review_requested"]


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


def test_question_progress_uses_paper_offset_and_records_acceptance() -> None:
    events = []
    results = QuestionGenerationGraph().generate(
        template=template(), chunks=chunks(), difficulty="Medium", bloom_level="Apply",
        candidate_count=2, question_offset=3,
        trace=lambda message, **kwargs: events.append({"message": message, **kwargs}),
    )
    assert len(results) == 2
    accepted = [event for event in events if event["stage"] == "question_accepted"]
    assert [event["details"]["question_index"] for event in accepted] == [4, 5]
    assert all(event["details"]["question_index"] in {4, 5} for event in events)


def test_review_timeout_does_not_repeat_network_calls_or_regenerate_candidate() -> None:
    import httpx

    class TimeoutReviewer:
        provider_name = "timeout-reviewer"

        def __init__(self):
            self.calls = []

        def generate_structured(self, prompt):
            self.calls.append(prompt.splitlines()[0])
            raise httpx.ReadTimeout("Review timed out")

    provider = TimeoutReviewer()
    graph = QuestionGenerationGraph(critic_provider=provider, max_attempts_per_question=5)
    events = []
    with pytest.raises(GenerationError, match="review service is unavailable"):
        graph.generate(
            template=template(), chunks=chunks(), difficulty="Medium", bloom_level="Apply",
            trace=lambda message, **kwargs: events.append({"message": message, **kwargs}),
        )
    assert sorted(provider.calls) == ["ROLE|INDEPENDENT_ANSWER_SOLVER", "ROLE|QUESTION_CRITIC"]
    assert len([event for event in events if event["stage"] == "preparing"]) == 1
    assert not any(event["stage"] == "candidate_retry" for event in events)
    assert any(event["stage"] == "review_unavailable" for event in events)
