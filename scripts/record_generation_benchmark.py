"""Run the real generation graph on development benchmark cases."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.api.questions import _configured_provider, _attach_failover_trace  # noqa: E402  # type: ignore[reportMissingImports]
from app.core.config import get_settings  # noqa: E402  # type: ignore[reportMissingImports]
from app.models import DocumentChunkDocument, QuestionTemplateDocument  # noqa: E402  # type: ignore[reportMissingImports]
from app.services.generation import GenerationError, ProviderRateLimitError  # noqa: E402  # type: ignore[reportMissingImports]
from app.services.question_agents import QuestionGenerationGraph  # noqa: E402  # type: ignore[reportMissingImports]


def _read_fixture(path: Path) -> dict[str, Any]:
    try:
        fixture = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read benchmark fixture '{path}': {error}") from error
    if not isinstance(fixture, dict) or not isinstance(fixture.get("cases"), list):
        raise SystemExit(f"Benchmark fixture '{path}' must contain a cases array.")
    return fixture


def _template(case: dict[str, Any]) -> QuestionTemplateDocument:
    request = case["request"]
    question_type = request["question_type"]
    required_fields = (
        ["question_text", "options", "correct_answer", "explanation"]
        if question_type.casefold() == "mcq"
        else ["question_text", "expected_answer", "explanation"]
    )
    return QuestionTemplateDocument(
        name=f"Benchmark {case['case_id']}",
        question_type=question_type,
        pattern=request["pattern"],
        required_fields=required_fields,
        supported_difficulties=[request["difficulty"]],
        supported_bloom_levels=[request["bloom_level"]],
        version="benchmark-v1",
        marks=1,
    )


def _context(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "chunk": DocumentChunkDocument(
                id=source["chunk_id"],
                study_material_id=f"benchmark-{case['case_id']}",
                chunk_index=index,
                page_number=source["page"],
                content=source["content"],
            ),
            "score": 1.0,
        }
        for index, source in enumerate(case["context"])
    ]


def record_development_outputs(
    fixture: dict[str, Any],
    *,
    provider: Any,
) -> list[dict[str, Any]]:
    if provider is None:
        raise ValueError("No hosted generation provider is configured.")

    outputs: list[dict[str, Any]] = []
    for case in fixture["cases"]:
        if case["split"] != "development":
            continue
        events: list[dict[str, Any]] = []

        def trace(
            message: str,
            *,
            stage: str | None = None,
            level: str = "info",
            duration_ms: float | None = None,
            details: dict[str, Any] | None = None,
            metric_deltas: dict[str, int | float] | None = None,
        ) -> None:
            event: dict[str, Any] = {"message": message, "level": level}
            if stage:
                event["stage"] = stage
            if duration_ms is not None:
                event["duration_ms"] = duration_ms
            if details:
                event["details"] = details
            if metric_deltas:
                event["metric_deltas"] = dict(metric_deltas)
            events.append(event)

        _attach_failover_trace(provider, trace)
        request = case["request"]
        started = perf_counter()
        try:
            questions = QuestionGenerationGraph(provider=provider).generate(
                template=_template(case),
                chunks=_context(case),
                difficulty=request["difficulty"],
                bloom_level=request["bloom_level"],
                candidate_count=1,
                trace=trace,
                run_llm_review=False,
            )
            decision_event = next(
                (event for event in reversed(events) if event.get("stage") == "jev_decision"),
                None,
            )
            if decision_event is None:
                raise GenerationError("Generation completed without a Jev decision.")
            decision = decision_event["details"]["decision"]
            route = next(
                (
                    event["message"].removeprefix("Successful model route: ").removesuffix(".")
                    for event in reversed(events)
                    if event.get("stage") == "model_route"
                ),
                getattr(provider, "provider_name", provider.__class__.__name__),
            )
            outputs.append({
                "case_id": case["case_id"],
                "status": "completed",
                "duration_ms": round((perf_counter() - started) * 1000),
                "model_route": route,
                "decision": decision,
                "question": questions[0].model_dump(),
            })
        except (GenerationError, ProviderRateLimitError) as error:
            outputs.append({
                "case_id": case["case_id"],
                "status": "failed",
                "duration_ms": round((perf_counter() - started) * 1000),
                "model_route": getattr(provider, "provider_name", provider.__class__.__name__),
                "error": f"{type(error).__name__}: {error}",
            })
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate real model outputs and Jev decisions for development benchmark cases only."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "storage" / "benchmarks" / "generation-development-results.json",
        help="Results JSON output path.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=ROOT / "backend" / "fixtures" / "generation_quality_benchmark.json",
        help="Benchmark fixture (defaults to the checked-in fixture).",
    )
    args = parser.parse_args()
    fixture = _read_fixture(args.fixture)
    settings = get_settings()
    provider = _configured_provider(settings)
    if provider is None:
        raise SystemExit("No hosted generation provider is configured; no benchmark requests were sent.")

    try:
        outputs = record_development_outputs(fixture, provider=provider)
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"Could not run the development benchmark: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    recorded_at = datetime.now(UTC).isoformat()
    for output in outputs:
        output["recorded_at"] = recorded_at
    args.output.write_text(
        json.dumps(outputs, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    completed = sum(output["status"] == "completed" for output in outputs)
    print(f"Recorded {completed}/{len(outputs)} development benchmark output(s) to '{args.output}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
