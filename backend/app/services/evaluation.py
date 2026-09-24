from __future__ import annotations

from collections import Counter
from typing import Any


class ResearchEvaluationService:
    def evaluate(self, questions: list[dict[str, Any]]) -> dict[str, Any]:
        total_questions = len(questions)
        if total_questions == 0:
            return {
                "total_questions": 0,
                "duplicate_rate": 0.0,
                "coverage": 0.0,
                "difficulty_distribution": {},
                "bloom_distribution": {},
                "pattern_distribution": {},
            }

        texts = [str(question.get("question_text", "")).strip().lower() for question in questions]
        duplicate_count = 0
        seen: set[str] = set()
        for text in texts:
            if text in seen:
                duplicate_count += 1
            seen.add(text)

        unique_texts = len(seen)
        duplicate_rate = round(duplicate_count / total_questions, 2) if total_questions else 0.0

        difficulty_distribution = dict(Counter(question.get("difficulty", "Unknown") for question in questions))
        bloom_distribution = dict(Counter(question.get("bloom_level", "Unknown") for question in questions))
        pattern_distribution = dict(Counter(question.get("pattern", "Unknown") for question in questions))

        unique_patterns = len(pattern_distribution)
        coverage = round(min(1.0, unique_patterns / max(1, len(pattern_distribution) or 1)), 2)
        if total_questions > 0:
            coverage = round(min(1.0, (unique_patterns + unique_texts) / (max(1, len(pattern_distribution)) + total_questions)), 2)

        return {
            "total_questions": total_questions,
            "duplicate_rate": duplicate_rate,
            "coverage": coverage,
            "difficulty_distribution": difficulty_distribution,
            "bloom_distribution": bloom_distribution,
            "pattern_distribution": pattern_distribution,
        }


def evaluate_question_set(questions: list[dict[str, Any]]) -> dict[str, Any]:
    return ResearchEvaluationService().evaluate(questions)


def run_research_evaluation(questions: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = evaluate_question_set(questions)
    return {
        "baseline": "study_material_plus_templates",
        "metrics": metrics,
    }
