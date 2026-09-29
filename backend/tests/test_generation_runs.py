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
