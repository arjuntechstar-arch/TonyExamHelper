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
