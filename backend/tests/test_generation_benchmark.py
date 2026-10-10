import json
from pathlib import Path
import subprocess
import sys

import pytest

from app.services.generation_benchmark import evaluate_generation_benchmark


def test_checked_in_generation_benchmark_has_disjoint_development_and_held_out_cases() -> None:
    fixture_path = Path(__file__).parents[1] / "fixtures" / "generation_quality_benchmark.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    cases = fixture["cases"]

    assert fixture["version"] == 1
    assert len({case["case_id"] for case in cases}) == len(cases) == 6
    assert [case["split"] for case in cases].count("development") == 4
    assert [case["split"] for case in cases].count("held_out") == 2
    for case in cases:
        available_chunks = {chunk["chunk_id"] for chunk in case["context"]}
        assert set(case["expected"]["evidence_chunk_ids"]) <= available_chunks


def test_generation_benchmark_cli_scores_recorded_results(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps([_output("dev-arrays-binary-search", "arrays-01")]),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_generation_benchmark.py"),
            "--results",
            str(results_path),
        ],
        cwd=root,
        capture_output=True,
        check=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["benchmark"] == "generation-quality-v1"
    assert report["case_count"] == 6
    assert report["output_count"] == 1
    assert report["splits"]["development"]["completed_count"] == 1
    assert report["splits"]["held_out"]["case_count"] == 2


def test_generation_benchmark_reviewer_cli_labels_development_only_and_preserves_source(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    source_path = tmp_path / "results.json"
    output_path = tmp_path / "reviewed.json"
    development = _output("dev-arrays-binary-search", "arrays-01")
    development["decision"] = {
        "action": "accept_locally",
        "review_required": False,
        "evidence_confidence": 0.92,
        "reasons": [],
    }
    held_out = _output("holdout-probability-bayes", "probability-06")
    held_out["decision"] = {
        "action": "review_with_hosted_checks",
        "review_required": True,
        "evidence_confidence": 0.42,
        "reasons": ["low_evidence_confidence"],
    }
    original_results = [development, held_out]
    source_path.write_text(json.dumps(original_results), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "review_generation_benchmark.py"),
            "--results",
            str(source_path),
            "--output",
            str(output_path),
            "--reviewer",
            "reviewer-1",
        ],
        cwd=root,
        capture_output=True,
        check=True,
        input="y\nSupported by the cited passage.\n",
        text=True,
    )

    reviewed = json.loads(output_path.read_text(encoding="utf-8"))
    untouched_source = json.loads(source_path.read_text(encoding="utf-8"))
    assert reviewed[0]["human_review"]["factual_correct"] is True
    assert reviewed[0]["human_review"]["reviewer"] == "reviewer-1"
    assert reviewed[0]["human_review"]["note"] == "Supported by the cited passage."
    assert "human_review" not in reviewed[1]
    assert "human_review" not in untouched_source[0]
    assert "human_review" not in untouched_source[1]
    assert "Development case: dev-arrays-binary-search" in completed.stdout
    assert "Development case: holdout-probability-bayes" not in completed.stdout
    assert "evidence_confidence" not in completed.stdout
    assert "held-out cases were not shown" in completed.stdout


def _case(case_id: str, split: str, chunk_id: str) -> dict:
    return {
        "case_id": case_id,
        "split": split,
        "request": {"question_type": "MCQ", "pattern": "Direct Concept"},
        "context": [{"chunk_id": chunk_id, "page": 2, "content": "Binary search discards half."}],
        "expected": {
            "question_type": "MCQ",
            "pattern": "Direct Concept",
            "difficulty": "Medium",
            "bloom_level": "Apply",
            "reference_answer": "Discard half of the search interval.",
            "evidence_chunk_ids": [chunk_id],
        },
    }


def _output(case_id: str, chunk_id: str, *, duration_ms: int = 100) -> dict:
    return {
        "case_id": case_id,
        "status": "completed",
        "duration_ms": duration_ms,
        "model_route": "deterministic-test",
        "question": {
            "question_text": "How does binary search narrow its search interval?",
            "options": [
                {"key": "A", "text": "Discard half of the search interval."},
                {"key": "B", "text": "It sorts the array repeatedly."},
                {"key": "C", "text": "It checks every pair of elements."},
                {"key": "D", "text": "It removes the middle element."},
            ],
            "correct_answer": "A",
            "explanation": "Each comparison discards half of the remaining interval.",
            "difficulty": "Medium",
            "bloom_level": "Apply",
            "sources": [{"chunk_id": chunk_id, "page": 2}],
            "question_type": "MCQ",
            "pattern": "Direct Concept",
            "marks": 1,
        },
    }


def test_generation_benchmark_reports_held_out_quality_and_latency_separately() -> None:
    cases = [
        _case("dev-one", "development", "chunk-one"),
        _case("dev-two", "development", "chunk-two"),
        _case("hold-one", "held_out", "chunk-held"),
    ]
    first = _output("dev-one", "chunk-one", duration_ms=100)
    first["human_review"] = {"factual_correct": True}
    first["decision"] = {
        "review_required": False,
        "evidence_confidence": 0.9,
        "reasons": [],
    }
    second = _output("dev-two", "chunk-two", duration_ms=300)
    second["human_review"] = {"factual_correct": False}
    second["decision"] = {
        "review_required": True,
        "evidence_confidence": 0.3,
        "reasons": ["low_evidence_confidence"],
    }
    held_out = _output("hold-one", "not-in-context", duration_ms=200)
    held_out["human_review"] = {"factual_correct": False}
    held_out["decision"] = {
        "review_required": True,
        "evidence_confidence": 0.1,
        "reasons": ["low_evidence_confidence"],
    }
    outputs = [
        first,
        second,
        held_out,
    ]

    result = evaluate_generation_benchmark(cases, outputs)

    development = result["splits"]["development"]
    held_out = result["splits"]["held_out"]
    assert result["case_count"] == 3
    assert development["metrics"]["source_grounding"] == 1.0
    assert development["metrics"]["answer_reference_f1"] == 1.0
    assert development["metrics"]["answer_evidence_token_precision"] > 0
    assert development["metrics"]["bloom_accuracy"] == 1.0
    assert development["metrics"]["difficulty_accuracy"] == 1.0
    assert development["metrics"]["format_compliance"] == 1.0
    assert development["duplicate_count"] == 1
    assert development["duplicate_rate"] == 0.5
    assert development["human_review"] == {
        "count": 2,
        "factual_correctness_rate": 0.5,
    }
    assert development["routing_review"] == {
        "labeled_decision_count": 2,
        "review_trigger_rate": 0.5,
        "factual_error_review_recall": 1.0,
        "factual_correct_review_rate": 0.0,
    }
    calibration = result["jev_threshold_calibration"]
    assert calibration["source_split"] == "development"
    assert calibration["labeled_decision_count"] == 2
    assert calibration["threshold_selected"] is None
    threshold_point = next(
        point for point in calibration["threshold_sweep"]
        if point["minimum_evidence_confidence"] == 0.6
    )
    assert threshold_point == {
        "minimum_evidence_confidence": 0.6,
        "reviewed_count": 1,
        "review_trigger_rate": 0.5,
        "factual_error_review_recall": 1.0,
        "factual_correct_review_rate": 0.0,
        "missed_factual_errors": 0,
    }
    assert development["latency"] == {
        "count": 2,
        "mean_ms": 200.0,
        "p50_ms": 100,
        "p95_ms": 300,
    }
    assert held_out["metrics"]["source_grounding"] == 0.0
    assert held_out["metrics"]["answer_reference_f1"] == 1.0
    assert held_out["metrics"]["answer_evidence_token_precision"] == 0.0
    assert held_out["duplicate_count"] == 0
    assert result["per_case"][2]["model_route"] == "deterministic-test"


def test_jev_threshold_sweep_preserves_other_review_reasons_and_excludes_unreviewable_rows() -> None:
    cases = [
        _case("dev-high-risk", "development", "chunk-one"),
        _case("dev-unknown-reason", "development", "chunk-two"),
    ]
    high_risk = _output("dev-high-risk", "chunk-one")
    high_risk["human_review"] = {"factual_correct": True}
    high_risk["decision"] = {
        "review_required": True,
        "evidence_confidence": 0.9,
        "reasons": ["advanced_bloom_level"],
    }
    unknown_reasons = _output("dev-unknown-reason", "chunk-two")
    unknown_reasons["human_review"] = {"factual_correct": False}
    unknown_reasons["decision"] = {
        "review_required": True,
        "evidence_confidence": 0.2,
    }

    calibration = evaluate_generation_benchmark(cases, [high_risk, unknown_reasons])[
        "jev_threshold_calibration"
    ]

    assert calibration["labeled_decision_count"] == 1
    assert calibration["excluded_missing_decision_reasons"] == 1
    assert all(point["reviewed_count"] == 1 for point in calibration["threshold_sweep"])


def test_jev_threshold_sweep_stays_empty_without_human_labeled_decisions() -> None:
    case = _case("dev-one", "development", "chunk-one")
    output = _output("dev-one", "chunk-one")
    output["decision"] = {
        "review_required": False,
        "evidence_confidence": 0.9,
        "reasons": [],
    }

    calibration = evaluate_generation_benchmark([case], [output])["jev_threshold_calibration"]

    assert calibration["labeled_decision_count"] == 0
    assert calibration["threshold_sweep"] == []
    assert calibration["threshold_selected"] is None


def test_generation_benchmark_detects_wrong_answer_bloom_level_and_format() -> None:
    case = _case("dev-one", "development", "chunk-one")
    output = _output("dev-one", "chunk-one")
    output["question"]["options"] = output["question"]["options"][:2]
    output["question"]["options"][0]["text"] = "Visit every array element."
    output["question"]["bloom_level"] = "Remember"
    output["question"]["difficulty"] = "Hard"

    result = evaluate_generation_benchmark([case], [output])
    metrics = result["splits"]["development"]["metrics"]

    assert metrics["answer_reference_f1"] < 1.0
    assert metrics["bloom_accuracy"] == 0.0
    assert metrics["difficulty_accuracy"] == 0.0
    assert metrics["format_compliance"] == 0.0
    assert "MCQ must contain exactly four unique option keys." in result["per_case"][0]["format_errors"]


def test_generation_benchmark_counts_failed_and_missing_outputs_without_fake_latency() -> None:
    cases = [
        _case("failed", "development", "chunk-one"),
        _case("missing", "held_out", "chunk-two"),
    ]
    outputs = [{
        "case_id": "failed",
        "status": "failed",
        "duration_ms": 250,
        "model_route": "hosted:test-model",
        "error": "provider timeout",
    }]

    result = evaluate_generation_benchmark(cases, outputs)

    assert result["splits"]["development"]["failed_count"] == 1
    assert result["splits"]["development"]["latency"]["mean_ms"] == 250
    assert result["splits"]["held_out"]["failed_count"] == 1
    assert result["splits"]["held_out"]["latency"] == {
        "count": 0,
        "mean_ms": None,
        "p50_ms": None,
        "p95_ms": None,
    }


@pytest.mark.parametrize(
    ("cases", "outputs", "message"),
    [
        ([], [], "At least one benchmark case"),
        ([_case("same", "development", "one"), _case("same", "held_out", "two")], [], "IDs must be unique"),
        ([_case("known", "development", "one")], [{
            "case_id": "unknown", "status": "failed", "duration_ms": 1,
        }], "unknown benchmark case"),
        ([_case("known", "development", "one")], [
            {"case_id": "known", "status": "failed", "duration_ms": 1},
            {"case_id": "known", "status": "failed", "duration_ms": 2},
        ], "Multiple outputs"),
        ([_case("known", "development", "one")], [{
            "case_id": "known", "status": "completed", "duration_ms": -1, "question": {},
        }], "non-negative duration_ms"),
        ([_case("known", "development", "one")], [{
            "case_id": "known", "status": "failed", "duration_ms": 1,
            "human_review": {"factual_correct": "yes"},
        }], "boolean factual_correct"),
        ([_case("known", "development", "one")], [{
            "case_id": "known", "status": "failed", "duration_ms": 1,
            "decision": {"review_required": True, "evidence_confidence": 1.5},
        }], "evidence_confidence from 0 to 1"),
    ],
)
def test_generation_benchmark_rejects_invalid_manifest_or_results(
    cases: list[dict],
    outputs: list[dict],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_generation_benchmark(cases, outputs)
