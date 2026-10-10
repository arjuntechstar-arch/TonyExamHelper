"""Collect blinded human factual-correctness labels for development outputs."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any, Callable

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.generation import GeneratedQuestion  # noqa: E402  # type: ignore[reportMissingImports]
from app.services.generation_benchmark import evaluate_generation_benchmark  # noqa: E402  # type: ignore[reportMissingImports]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read JSON file '{path}': {error}") from error


def _write_json(path: Path, value: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError as error:
        raise SystemExit(f"Could not write reviewed results to '{path}': {error}") from error


def _answer(question: dict[str, Any]) -> str:
    correct_key = question.get("correct_answer")
    option = next(
        (item.get("text", "") for item in question.get("options", []) if item.get("key") == correct_key),
        "",
    )
    return str(option or question.get("expected_answer") or "(No answer supplied)")


def _review_prompt(read: Callable[[str], str], write: Callable[[str], None]) -> bool | None:
    while True:
        answer = read("Is the generated answer factually correct and supported by the cited evidence? [y/n/s]: ")
        normalized = answer.strip().casefold()
        if normalized in {"y", "yes"}:
            return True
        if normalized in {"n", "no"}:
            return False
        if normalized in {"s", "skip"}:
            return None
        write("Enter y (correct), n (incorrect), or s (skip).")


def _has_reviewable_decision(output: dict[str, Any] | None) -> bool:
    if output is None or output["status"] != "completed":
        return False
    decision = output.get("decision")
    if not isinstance(decision, dict):
        return False
    reasons = decision.get("reasons")
    return isinstance(reasons, list) and all(isinstance(reason, str) for reason in reasons)


def review_development_outputs(
    cases: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    *,
    reviewer: str,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    if not reviewer.strip():
        raise ValueError("Reviewer identity must not be blank.")

    result_by_id = {item["case_id"]: item for item in outputs}
    development_cases = [case for case in cases if case["split"] == "development"]
    eligible_count = 0
    for case in development_cases:
        output = result_by_id.get(case["case_id"])
        if not _has_reviewable_decision(output):
            continue
        assert output is not None
        if output.get("human_review", {}).get("factual_correct") in (True, False):
            continue
        try:
            GeneratedQuestion.model_validate(output.get("question"))
        except ValidationError:
            continue
        eligible_count += 1

    if eligible_count == 0:
        write("No unlabeled completed development outputs with Jev decisions are available to review.")
        return outputs

    reviewed_at = datetime.now(UTC).isoformat()
    completed = 0
    for case in development_cases:
        output = result_by_id.get(case["case_id"])
        if not _has_reviewable_decision(output):
            continue
        assert output is not None
        if output.get("human_review", {}).get("factual_correct") in (True, False):
            continue
        try:
            question = GeneratedQuestion.model_validate(output.get("question")).model_dump()
        except ValidationError:
            continue

        write(f"\nDevelopment case: {case['case_id']}")
        write(f"Question: {question.get('question_text', '(No question text supplied)')}")
        write(f"Answer: {_answer(question)}")
        if question.get("explanation"):
            write(f"Explanation: {question['explanation']}")

        contexts = {item["chunk_id"]: item for item in case["context"]}
        sources = question.get("sources", [])
        for source in sources:
            passage = contexts.get(source.get("chunk_id"))
            if passage:
                write(
                    f"Evidence [{source['chunk_id']}, page {passage['page']}]: "
                    f"{passage['content']}"
                )
            else:
                write(
                    f"Evidence [{source.get('chunk_id', 'unknown')}, "
                    f"page {source.get('page', 'unknown')}]: not present in benchmark context."
                )
        if not sources:
            write("Evidence: the generated question cites no source passage.")

        factual_correct = _review_prompt(read, write)
        if factual_correct is None:
            continue
        note = read("Optional review note (press Enter to leave blank): ").strip()
        review: dict[str, Any] = {
            "factual_correct": factual_correct,
            "reviewer": reviewer.strip(),
            "reviewed_at": reviewed_at,
        }
        if note:
            review["note"] = note
        output["human_review"] = review
        completed += 1

    write(f"\nSaved {completed} human factual-correctness label(s); held-out cases were not shown.")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Blindly review generated answers against cited evidence for benchmark development cases."
    )
    parser.add_argument("--results", required=True, type=Path, help="Recorded benchmark output JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Path for reviewed JSON (must differ from --results).")
    parser.add_argument("--reviewer", required=True, help="Reviewer name or internal identifier for audit metadata.")
    parser.add_argument(
        "--fixture",
        default=ROOT / "backend" / "fixtures" / "generation_quality_benchmark.json",
        type=Path,
        help="Benchmark manifest (defaults to the checked-in fixture).",
    )
    args = parser.parse_args()
    if args.results.resolve() == args.output.resolve():
        raise SystemExit("--output must be a different path from --results; source results are never overwritten.")

    fixture = _read_json(args.fixture)
    results = _read_json(args.results)
    if not isinstance(fixture, dict) or not isinstance(fixture.get("cases"), list):
        raise SystemExit(f"Benchmark fixture '{args.fixture}' must contain a cases array.")
    if not isinstance(results, list):
        raise SystemExit(f"Results file '{args.results}' must contain a JSON array.")
    try:
        evaluate_generation_benchmark(fixture["cases"], results)
        reviewed = review_development_outputs(
            fixture["cases"],
            results,
            reviewer=args.reviewer,
        )
        evaluate_generation_benchmark(fixture["cases"], reviewed)
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"Could not review generation results: {error}") from error
    _write_json(args.output, reviewed)
    print(f"Reviewed results written to '{args.output}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
