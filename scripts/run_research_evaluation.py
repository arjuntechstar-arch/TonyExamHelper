"""Run the fixed Phase 13 benchmark without network or database access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.evaluation import run_research_evaluation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the repeatable research benchmark.")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=ROOT / "backend" / "fixtures" / "research_benchmark.json",
    )
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    result = run_research_evaluation(
        fixture["questions"],
        expected_topics=fixture["expected_topics"],
        baseline_questions=fixture.get("baseline_questions"),
    )
    print(json.dumps({"benchmark": fixture["name"], **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
