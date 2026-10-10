from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

from app.services.assessment_schema import is_mcq_question_type
from app.services.generation import GeneratedQuestion
from app.services.quality import QualityConfig, semantic_similarity, semantic_tokens


def _tokens(value: str) -> Counter[str]:
    return Counter(re.findall(r"[a-z0-9]+", value.casefold()))


def _answer_text(question: GeneratedQuestion) -> str:
    if question.correct_answer:
        option = next(
            (option.text for option in question.options if option.key == question.correct_answer),
            "",
        )
        if option:
            return option
    return question.expected_answer or ""


def _reference_f1(reference: str, answer: str) -> float:
    reference_tokens = _tokens(reference)
    answer_tokens = _tokens(answer)
    if not reference_tokens or not answer_tokens:
        return 0.0
    overlap = sum((reference_tokens & answer_tokens).values())
    precision = overlap / sum(answer_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _evidence_token_precision(answer: str, passages: list[str]) -> float:
    answer_tokens = semantic_tokens(answer)
    evidence_tokens = set().union(*(semantic_tokens(passage) for passage in passages))
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & evidence_tokens) / len(answer_tokens)


def _format_errors(question: GeneratedQuestion, expected: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if question.question_type.casefold() != str(expected["question_type"]).casefold():
        errors.append("Question type does not match the benchmark request.")
    if question.pattern.casefold() != str(expected["pattern"]).casefold():
        errors.append("Question pattern does not match the benchmark request.")
    if is_mcq_question_type(question.question_type):
        keys = [option.key for option in question.options]
        if len(keys) != 4 or len(set(keys)) != 4:
            errors.append("MCQ must contain exactly four unique option keys.")
        if question.correct_answer not in keys:
            errors.append("MCQ correct_answer must reference one of its option keys.")
        if question.expected_answer:
            errors.append("MCQ must not include a written expected_answer.")
    else:
        if question.options:
            errors.append("Written-response questions must not contain MCQ options.")
        if question.correct_answer is not None:
            errors.append("Written-response questions must not use an option-key correct_answer.")
        if not (question.expected_answer or "").strip():
            errors.append("Written-response questions require an expected_answer.")
    return errors


def _evaluate_case(
    case: dict[str, Any],
    output: dict[str, Any] | None,
) -> dict[str, Any]:
    expected = case["expected"]
    result: dict[str, Any] = {
        "case_id": case["case_id"],
        "split": case["split"],
        "status": "missing" if output is None else output["status"],
        "duration_ms": None if output is None else output["duration_ms"],
        "model_route": None if output is None else output.get("model_route"),
        "human_factual_correctness": (
            output.get("human_review", {}).get("factual_correct")
            if output is not None and output.get("human_review") is not None
            else None
        ),
        "decision": output.get("decision") if output is not None else None,
        "source_grounding": 0.0,
        "answer_reference_f1": 0.0,
        "answer_evidence_token_precision": 0.0,
        "bloom_accuracy": 0.0,
        "difficulty_accuracy": 0.0,
        "format_compliance": 0.0,
        "format_errors": [],
        "_question": None,
    }
    if output is None or output["status"] != "completed":
        return result

    try:
        question = GeneratedQuestion.model_validate(output.get("question"))
    except ValidationError as error:
        result["format_errors"] = [
            f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}"
            for issue in error.errors()
        ]
        return result
    except (TypeError, ValueError) as error:
        result["format_errors"] = [str(error)]
        return result

    result["_question"] = question
    format_errors = _format_errors(question, expected)
    result["format_errors"] = format_errors
    context_pages = {
        chunk["chunk_id"]: chunk["page"]
        for chunk in case["context"]
    }
    context_content = {
        chunk["chunk_id"]: chunk["content"]
        for chunk in case["context"]
    }
    referenced_sources = {source.chunk_id for source in question.sources}
    expected_evidence = set(expected["evidence_chunk_ids"])
    result["source_grounding"] = float(
        bool(referenced_sources)
        and referenced_sources.issubset(context_pages)
        and bool(referenced_sources & expected_evidence)
        and all(context_pages[source.chunk_id] == source.page for source in question.sources)
    )
    result["answer_reference_f1"] = round(
        _reference_f1(expected["reference_answer"], _answer_text(question)),
        4,
    )
    result["answer_evidence_token_precision"] = round(
        _evidence_token_precision(
            _answer_text(question),
            [
                context_content[source_id]
                for source_id in referenced_sources
                if source_id in context_content
            ],
        ),
        4,
    )
    result["bloom_accuracy"] = float(
        question.bloom_level.casefold() == str(expected["bloom_level"]).casefold()
    )
    result["difficulty_accuracy"] = float(
        question.difficulty.casefold() == str(expected["difficulty"]).casefold()
    )
    result["format_compliance"] = float(not format_errors)
    return result


def _latency_summary(values: Iterable[int]) -> dict[str, int | float | None]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    return {
        "count": len(ordered),
        "mean_ms": round(sum(ordered) / len(ordered), 2),
        "p50_ms": ordered[math.ceil(0.50 * len(ordered)) - 1],
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
    }


def _summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    metric_names = (
        "source_grounding",
        "answer_reference_f1",
        "answer_evidence_token_precision",
        "bloom_accuracy",
        "difficulty_accuracy",
        "format_compliance",
    )
    summary: dict[str, Any] = {
        "case_count": count,
        "completed_count": sum(row["status"] == "completed" for row in rows),
        "failed_count": sum(row["status"] in {"failed", "missing"} for row in rows),
        "metrics": {
            name: round(sum(float(row[name]) for row in rows) / count, 4) if count else 0.0
            for name in metric_names
        },
        "latency": _latency_summary(
            row["duration_ms"] for row in rows if row["duration_ms"] is not None
        ),
    }
    human_reviews = [
        row["human_factual_correctness"]
        for row in rows
        if row["human_factual_correctness"] is not None
    ]
    summary["human_review"] = {
        "count": len(human_reviews),
        "factual_correctness_rate": (
            round(sum(human_reviews) / len(human_reviews), 4)
            if human_reviews
            else None
        ),
    }
    reviewed_decisions = [
        row for row in rows
        if row["decision"] is not None and row["human_factual_correctness"] is not None
    ]
    incorrect_decisions = [
        row for row in reviewed_decisions
        if not row["human_factual_correctness"]
    ]
    correct_decisions = [
        row for row in reviewed_decisions
        if row["human_factual_correctness"]
    ]
    summary["routing_review"] = {
        "labeled_decision_count": len(reviewed_decisions),
        "review_trigger_rate": (
            round(sum(row["decision"]["review_required"] for row in reviewed_decisions) / len(reviewed_decisions), 4)
            if reviewed_decisions
            else None
        ),
        "factual_error_review_recall": (
            round(sum(row["decision"]["review_required"] for row in incorrect_decisions) / len(incorrect_decisions), 4)
            if incorrect_decisions
            else None
        ),
        "factual_correct_review_rate": (
            round(sum(row["decision"]["review_required"] for row in correct_decisions) / len(correct_decisions), 4)
            if correct_decisions
            else None
        ),
    }
    duplicate_ids: set[str] = set()
    duplicate_threshold = QualityConfig().duplicate_threshold
    valid_questions = [
        row for row in rows
        if isinstance(row["_question"], GeneratedQuestion)
    ]
    for index, row in enumerate(valid_questions):
        if any(
            semantic_similarity(row["_question"], previous["_question"]) >= duplicate_threshold
            for previous in valid_questions[:index]
        ):
            duplicate_ids.add(row["case_id"])
    summary["duplicate_count"] = len(duplicate_ids)
    summary["duplicate_rate"] = round(len(duplicate_ids) / count, 4) if count else 0.0
    return summary


def _jev_threshold_calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled_decisions = [
        row
        for row in rows
        if row["decision"] is not None
        and row["human_factual_correctness"] is not None
        and isinstance(row["decision"].get("reasons"), list)
        and all(isinstance(reason, str) for reason in row["decision"]["reasons"])
    ]
    labeled_without_reasons = sum(
        row["decision"] is not None
        and row["human_factual_correctness"] is not None
        and (
            not isinstance(row["decision"].get("reasons"), list)
            or not all(isinstance(reason, str) for reason in row["decision"]["reasons"])
        )
        for row in rows
    )
    confidences = sorted({float(row["decision"]["evidence_confidence"]) for row in labeled_decisions})
    thresholds = {0.0, 1.0, *confidences} if labeled_decisions else set()
    thresholds.update(
        round((lower + upper) / 2, 6)
        for lower, upper in zip(confidences, confidences[1:])
    )
    incorrect_count = sum(not row["human_factual_correctness"] for row in labeled_decisions)
    correct_count = len(labeled_decisions) - incorrect_count
    sweep = []
    for threshold in sorted(thresholds):
        reviewed = [
            row for row in labeled_decisions
            if any(reason != "low_evidence_confidence" for reason in row["decision"]["reasons"])
            or row["decision"]["evidence_confidence"] < threshold
        ]
        reviewed_incorrect = sum(not row["human_factual_correctness"] for row in reviewed)
        reviewed_correct = len(reviewed) - reviewed_incorrect
        sweep.append({
            "minimum_evidence_confidence": threshold,
            "reviewed_count": len(reviewed),
            "review_trigger_rate": round(len(reviewed) / len(labeled_decisions), 4) if labeled_decisions else None,
            "factual_error_review_recall": (
                round(reviewed_incorrect / incorrect_count, 4)
                if incorrect_count
                else None
            ),
            "factual_correct_review_rate": (
                round(reviewed_correct / correct_count, 4)
                if correct_count
                else None
            ),
            "missed_factual_errors": incorrect_count - reviewed_incorrect,
        })
    return {
        "source_split": "development",
        "labeled_decision_count": len(labeled_decisions),
        "excluded_missing_decision_reasons": labeled_without_reasons,
        "threshold_sweep": sweep,
        "threshold_selected": None,
    }


def evaluate_generation_benchmark(
    cases: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score model outputs against a versioned, human-authored benchmark manifest.

    Answer-reference F1 is a lexical agreement signal, not a substitute for
    expert review of factual correctness.
    """
    if not cases:
        raise ValueError("At least one benchmark case is required.")

    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Benchmark case IDs must be unique.")
    if any(case["split"] not in {"development", "held_out"} for case in cases):
        raise ValueError("Benchmark split must be 'development' or 'held_out'.")

    output_by_id: dict[str, dict[str, Any]] = {}
    known_ids = set(case_ids)
    for output in outputs:
        case_id = output["case_id"]
        if case_id not in known_ids:
            raise ValueError(f"Output references unknown benchmark case '{case_id}'.")
        if case_id in output_by_id:
            raise ValueError(f"Multiple outputs were provided for benchmark case '{case_id}'.")
        if output["status"] not in {"completed", "failed"}:
            raise ValueError(f"Output for '{case_id}' must have completed or failed status.")
        duration_ms = output.get("duration_ms")
        if not isinstance(duration_ms, int) or isinstance(duration_ms, bool) or duration_ms < 0:
            raise ValueError(f"Output for '{case_id}' must include a non-negative duration_ms.")
        if output["status"] == "completed" and not isinstance(output.get("question"), dict):
            raise ValueError(f"Completed output for '{case_id}' must include a question object.")
        review = output.get("human_review")
        if review is not None and (
            not isinstance(review, dict)
            or not isinstance(review.get("factual_correct"), bool)
        ):
            raise ValueError(
                f"Human review for '{case_id}' must include a boolean factual_correct label."
            )
        decision = output.get("decision")
        if decision is not None and (
            not isinstance(decision, dict)
            or not isinstance(decision.get("review_required"), bool)
            or not isinstance(decision.get("evidence_confidence"), (int, float))
            or isinstance(decision.get("evidence_confidence"), bool)
            or not 0 <= decision["evidence_confidence"] <= 1
        ):
            raise ValueError(
                f"Decision for '{case_id}' must include review_required and evidence_confidence from 0 to 1."
            )
        output_by_id[case_id] = output

    rows = [_evaluate_case(case, output_by_id.get(case["case_id"])) for case in cases]
    split_summaries = {
        split: _summarize_split([row for row in rows if row["split"] == split])
        for split in ("development", "held_out")
    }
    development_rows = [row for row in rows if row["split"] == "development"]
    return {
        "benchmark": "generation-quality",
        "version": 1,
        "case_count": len(cases),
        "output_count": len(outputs),
        "splits": split_summaries,
        "jev_threshold_calibration": _jev_threshold_calibration(development_rows),
        "per_case": [
            {key: value for key, value in row.items() if key != "_question"}
            for row in rows
        ],
        "metric_notes": {
            "answer_reference_f1": "Lexical overlap with the authored reference answer; expert review is still needed to establish correctness.",
            "answer_evidence_token_precision": "Share of answer content tokens found in the cited passage; this lexical support signal does not establish entailment.",
            "source_grounding": "Passes when cited chunk IDs and pages match the case evidence and at least one expected evidence chunk is cited.",
            "duplicate_rate": "Share of benchmark cases whose generated questions duplicate an earlier valid question in the same split.",
            "latency": "Summarized from supplied per-output duration_ms values, including failed calls; missing outputs have no measured latency.",
        },
    }
