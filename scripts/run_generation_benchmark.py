"""Score recorded generation outputs against the fixed quality benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.generation_benchmark import evaluate_generation_benchmark  # noqa: E402  # type: ignore[reportMissingImports]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read JSON file '{path}': {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate recorded question-generation outputs without making model calls."
    )
    parser.add_argument(
        "--results",
        required=True,
        type=Path,
        help="JSON file containing one completed or failed output per benchmark case.",
    )
    parser.add_argument(
        "--fixture",
        default=ROOT / "backend" / "fixtures" / "generation_quality_benchmark.json",
        type=Path,
        help="Versioned benchmark manifest (defaults to the checked-in benchmark).",
    )
    args = parser.parse_args()

    fixture = _read_json(args.fixture)
    results = _read_json(args.results)
    if not isinstance(fixture, dict) or not isinstance(fixture.get("cases"), list):
        raise SystemExit(f"Benchmark fixture '{args.fixture}' must contain a cases array.")
    if not isinstance(results, list):
        raise SystemExit(f"Results file '{args.results}' must contain a JSON array.")
    try:
        report = evaluate_generation_benchmark(fixture["cases"], results)
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"Could not evaluate generation results: {error}") from error
    report["benchmark"] = fixture.get("name", report["benchmark"])
    report["fixture_version"] = fixture.get("version")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
