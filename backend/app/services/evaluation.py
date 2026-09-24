from __future__ import annotations

from collections import Counter
from typing import Any


def _distribution(questions: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(question.get(key, "Unknown")) for question in questions))


def evaluate_question_set(
    questions: list[dict[str, Any]],
    *,
    expected_topics: list[str] | None = None,
) -> dict[str, Any]:
    """Calculate deterministic, model-independent quality metrics.

    ``expected_topics`` is supplied by the benchmark fixture (or a caller that
    has syllabus metadata). Without it, coverage is the share of questions
    carrying a non-empty topic, or distinct patterns when topic metadata is
    unavailable, preserving the original API's useful fallback.
    """
    total_questions = len(questions)
    normalized = [
        str(question.get("question_text", "")).strip().casefold()
        for question in questions
    ]
    unique_texts = len(set(normalized))
    duplicate_count = total_questions - unique_texts
    duplicate_rate = round(duplicate_count / total_questions, 2) if total_questions else 0.0

    topics = {
        str(question.get("topic", "")).strip()
        for question in questions
        if str(question.get("topic", "")).strip()
    }
    if expected_topics:
        expected = {topic.strip() for topic in expected_topics if topic.strip()}
        coverage = round(len(topics & expected) / len(expected), 2) if expected else 0.0
    else:
        # Older callers do not provide syllabus topics. Keep their coverage
        # signal useful by using distinct patterns as the fallback denominator.
        pattern_count = len(
            {
                str(question.get("pattern", "Unknown"))
                for question in questions
            }
        )
        coverage = round(
            (len(topics) if topics else pattern_count) / total_questions, 2
        ) if total_questions else 0.0

    return {
        "total_questions": total_questions,
        "unique_questions": unique_texts,
        "duplicate_count": duplicate_count,
        "duplicate_rate": duplicate_rate,
        "coverage": coverage,
        "covered_topics": sorted(topics),
        "difficulty_distribution": _distribution(questions, "difficulty"),
        "bloom_distribution": _distribution(questions, "bloom_level"),
        "pattern_distribution": _distribution(questions, "pattern"),
        "topic_distribution": _distribution(questions, "topic"),
    }


class ResearchEvaluationService:
    """Compatibility façade for callers that previously instantiated the service."""

    def evaluate(
        self,
        questions: list[dict[str, Any]],
        *,
        expected_topics: list[str] | None = None,
    ) -> dict[str, Any]:
        return evaluate_question_set(questions, expected_topics=expected_topics)


def run_research_evaluation(
    questions: list[dict[str, Any]],
    *,
    expected_topics: list[str] | None = None,
    baseline_questions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate a run and optionally compare it with a deterministic baseline."""
    result: dict[str, Any] = {
        "baseline": "study_material_plus_templates",
        "metrics": evaluate_question_set(questions, expected_topics=expected_topics),
    }
    if baseline_questions is not None:
        result["comparison"] = {
            "baseline_metrics": evaluate_question_set(
                baseline_questions, expected_topics=expected_topics
            ),
            "metric_delta": {
                key: round(
                    result["metrics"].get(key, 0) - evaluate_question_set(
                        baseline_questions, expected_topics=expected_topics
                    ).get(key, 0),
                    2,
                )
                for key in ("coverage", "duplicate_rate")
            },
        }
    return result
