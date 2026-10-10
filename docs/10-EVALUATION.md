# Research Evaluation

Baseline A:
Study Material → Plain LLM → Questions

Baseline B:
Study Material + Templates → LLM → Questions

Proposed:
Syllabus + Study Material + RAG + Templates + LLM + Validation → Questions

Metrics:
- relevance
- correctness
- pattern compliance
- difficulty accuracy
- Bloom accuracy
- syllabus coverage
- diversity
- duplicate rate
- generation time
- cost where applicable

Expert evaluation can rate relevance, correctness, clarity, difficulty, pattern compliance and educational value.

If a student study is conducted, obtain appropriate institutional approval and distinguish perceived preparedness from measured learning outcomes.

## Repeatable Phase 13 benchmark

The checked-in `backend/fixtures/research_benchmark.json` is a small, fixed,
offline fixture. It records expected syllabus topics, a proposed question set,
and a baseline set. Run it from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\run_research_evaluation.py
```

The JSON output includes total and unique questions, duplicate count/rate,
topic coverage, and difficulty/Bloom/pattern/topic distributions. When a
baseline is present it also reports baseline metrics and deltas for coverage
and duplicate rate. Since the fixture and evaluator are deterministic, the
command is suitable for regression checks without an LLM, database, or network.

## Generation quality benchmark (Phase 3)

`backend/fixtures/generation_quality_benchmark.json` contains four development
cases and two held-out cases. Each case records a representative request,
source passages with stable chunk IDs and page numbers, and human-authored
reference criteria. The held-out split is reported separately and should not
be used to tune prompts or thresholds.

The benchmark runner scores recorded outputs; it does not call an LLM or
fabricate latency values. Supply one JSON result per attempted case, including
failed calls:

```json
[
  {
    "case_id": "dev-arrays-binary-search",
    "status": "completed",
    "duration_ms": 842,
    "model_route": "provider:model",
    "human_review": {"factual_correct": true},
    "question": {
      "question_text": "How does binary search reduce its search interval?",
      "options": [
        {"key": "A", "text": "Discard half of the interval after comparing its middle element."},
        {"key": "B", "text": "Visit every pair of elements."},
        {"key": "C", "text": "Sort the array at every step."},
        {"key": "D", "text": "Remove an arbitrary array element."}
      ],
      "correct_answer": "A",
      "explanation": "The comparison identifies which half cannot contain the target.",
      "difficulty": "Medium",
      "bloom_level": "Apply",
      "sources": [{"chunk_id": "arrays-01", "page": 4}],
      "question_type": "MCQ",
      "pattern": "Scenario Based",
      "marks": 1
    }
  },
  {
    "case_id": "holdout-os-deadlock",
    "status": "failed",
    "duration_ms": 1000,
    "model_route": "provider:model",
    "error": "provider timeout"
  }
]
```

Run the evaluator from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\run_generation_benchmark.py --results path\to\recorded-results.json
```

To record fresh outputs from the configured hosted generation model and Jev
policy for development cases only, run:

```powershell
.\.venv\Scripts\python.exe scripts\record_generation_benchmark.py
```

This makes real provider requests and writes results to
`storage/benchmarks/generation-development-results.json` by default. It does
not run held-out cases. Use `--output` to select a different results path.

To collect factual labels without hand-editing JSON, run the blinded reviewer
against a recorded results file. It displays generated questions, answers,
explanations, and cited benchmark passages for development cases only; it does
not reveal Jev confidence or reasons before the label is entered. Each answer
can be marked correct, incorrect, or skipped, with an optional note. The
reviewer identity and review timestamp are recorded. The source results file
is never overwritten:

```powershell
.\.venv\Scripts\python.exe scripts\review_generation_benchmark.py `
  --results path\to\recorded-results.json `
  --output path\to\reviewed-results.json `
  --reviewer reviewer-id
```

Run `run_generation_benchmark.py` with the reviewed output file to see the
development-only Jev threshold sweep. Human labels must be made by a qualified
reviewer against the cited evidence; this tool does not infer or generate
labels.

The report gives development and held-out source attribution, answer-reference
F1, answer/evidence token precision, Bloom and difficulty exact-match accuracy,
structural format compliance, duplicate rate, and mean/p50/p95 latency. Missing
outputs count as failures but have no latency measurement; failed calls
contribute their recorded duration. Answer-reference F1 and answer/evidence
token precision are lexical signals, not proof of factual correctness. Expert
review remains necessary for factual correctness and educational value. Add
`human_review.factual_correct` only after a reviewer has checked the answer
against its cited passage; the report includes accuracy over those reviewed
cases separately and leaves it unavailable when no labels were supplied.
This small seed set should be expanded with reviewed examples before using its
scores to make model or routing decisions.

To measure the Phase 4 Jev routing policy, include the actual decision recorded
for each output:

```json
"decision": {
  "action": "review_with_hosted_checks",
  "review_required": true,
  "evidence_confidence": 0.42,
  "reasons": ["low_evidence_confidence"]
}
```

When both a decision and a human correctness label are present, each split
reports `routing_review.review_trigger_rate`,
`routing_review.factual_error_review_recall`, and
`routing_review.factual_correct_review_rate`. These measure review workload
and detection of factual errors; the evidence-confidence score is not
calibrated as a probability. Tune thresholds on reviewed development cases,
then report final performance on held-out cases without adjusting the policy.

The report's `jev_threshold_calibration` section sweeps candidate confidence
thresholds using only development cases that have both a human factual label
and a recorded decision with its `reasons`. It reports review workload,
factual-error recall, correct-answer review rate, and missed factual errors.
Non-confidence reasons such as `advanced_bloom_level` remain escalated during
the sweep. The tool never selects or applies a threshold: `threshold_selected`
is always `null`, and the sweep is empty until eligible human-reviewed
development outputs are supplied. Reviewers should choose a threshold based
on an agreed risk/workload target, then evaluate the frozen policy on held-out
cases without tuning against those results.
