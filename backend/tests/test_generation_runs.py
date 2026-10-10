from app.services.generation_runs import GenerationRunStore


def test_generation_run_records_structured_trace_and_exact_failure() -> None:
    store = GenerationRunStore()
    run = store.begin(request_type="single", user_id="user-1")
    run.log("Retrieved 5 source chunks.", stage="retrieval")
    run.log("Question validation: source_copy.", stage="question_validation", level="error")
    store.fail(run, "Question generation produced 0 of 1 required candidates: source_copy")

    snapshot = store.get(run.id).snapshot()  # type: ignore[union-attr]
    assert snapshot["status"] == "failed"
    assert snapshot["error"] == "Question generation produced 0 of 1 required candidates: source_copy"
    assert snapshot["logs"][-2]["stage"] == "question_validation"
    assert snapshot["logs"][-2]["level"] == "error"
    assert store.list(user_id="user-1")[0].id == run.id
    assert store.list(user_id="another-user") == []


def test_generation_run_records_durations_and_model_metrics() -> None:
    store = GenerationRunStore()
    run = store.begin(request_type="single", user_id="user-1")
    run.log(
        "Generation model call took 1.25s.",
        stage="generation_model",
        duration_ms=1250,
        metric_deltas={
            "model_calls": 1,
            "generation_calls": 1,
            "model_duration_ms": 1250,
            "generation_duration_ms": 1250,
        },
    )
    run.log(
        "Retrieved 5 chunks in 0.02s.",
        stage="retrieval",
        duration_ms=20,
        metric_deltas={"retrieval_duration_ms": 20},
        details={"decision": {
            "action": "review_with_hosted_checks",
            "review_required": True,
            "evidence_confidence": 0.42,
            "reasons": ["low_evidence_confidence"],
        }},
    )
    store.complete(run, [])

    snapshot = run.snapshot()
    assert snapshot["logs"][1]["duration_ms"] == 1250
    assert snapshot["metrics"]["model_calls"] == 1
    assert snapshot["metrics"]["generation_duration_ms"] == 1250
    assert snapshot["metrics"]["retrieval_duration_ms"] == 20
    assert snapshot["metrics"]["total_duration_ms"] >= 0
    assert snapshot["logs"][2]["details"]["decision"]["evidence_confidence"] == 0.42
