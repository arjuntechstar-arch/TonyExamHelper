from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from uuid import uuid4

logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="question-generation")


@dataclass
class GenerationRun:
    id: str = field(default_factory=lambda: str(uuid4()))
    status: str = "queued"
    stage: str = "queued"
    message: str = "Generation queued."
    logs: list[str] = field(default_factory=list)
    result: list[dict] | None = None
    error: str | None = None
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def log(self, message: str, *, stage: str | None = None) -> None:
        if stage:
            self.stage = stage
        self.message = message
        self.logs.append(message)
        self.updated_at = datetime.now(UTC).isoformat()
        logger.info("generation_run=%s stage=%s message=%s", self.id, self.stage, message)

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "logs": list(self.logs),
            "result": self.result,
            "error": self.error,
            "updated_at": self.updated_at,
        }


class GenerationRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, GenerationRun] = {}
        self._lock = Lock()

    def create(self, worker, *args, **kwargs) -> GenerationRun:
        run = GenerationRun()
        with self._lock:
            self._runs[run.id] = run
        executor.submit(self._execute, run, worker, *args, **kwargs)
        return run

    def get(self, run_id: str) -> GenerationRun | None:
        with self._lock:
            return self._runs.get(run_id)

    @staticmethod
    def _execute(run: GenerationRun, worker, *args, **kwargs) -> None:
        run.status = "running"
        run.log("Generation worker started.", stage="started")
        try:
            run.result = worker(run, *args, **kwargs)
            run.status = "completed"
            run.log("Generation completed successfully.", stage="completed")
        except Exception as error:
            run.status = "failed"
            run.error = str(error)
            run.log(f"Generation failed: {error}", stage="failed")
            logger.exception("Generation run %s failed", run.id)


generation_runs = GenerationRunStore()
