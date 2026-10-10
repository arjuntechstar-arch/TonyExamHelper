from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from time import monotonic
from typing import Callable
from uuid import uuid4

logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="question-generation")


@dataclass
class GenerationRun:
    id: str = field(default_factory=lambda: str(uuid4()))
    status: str = "queued"
    stage: str = "queued"
    message: str = "Generation queued."
    request_type: str = "paper"
    user_id: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    logs: list[dict] = field(default_factory=list)
    result: list[dict] | None = None
    error: str | None = None
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metrics: dict[str, int | float] = field(default_factory=dict)
    started_monotonic: float = field(default_factory=monotonic, repr=False, compare=False)
    on_update: Callable[["GenerationRun"], None] | None = field(default=None, repr=False, compare=False)
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def log(
        self,
        message: str,
        *,
        stage: str | None = None,
        level: str = "info",
        duration_ms: float | None = None,
        metric_deltas: dict[str, int | float] | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        with self._lock:
            if stage:
                self.stage = stage
            self.message = message
            self.updated_at = datetime.now(UTC).isoformat()
            event: dict[str, object] = {
                "timestamp": self.updated_at,
                "stage": self.stage,
                "level": level,
                "message": message,
            }
            if duration_ms is not None:
                event["duration_ms"] = round(duration_ms, 2)
            if details:
                event["details"] = dict(details)
            if metric_deltas:
                for key, value in metric_deltas.items():
                    self.metrics[key] = self.metrics.get(key, 0) + value
            self.logs.append(event)
            current_stage = self.stage
        logger.info("generation_run=%s stage=%s message=%s", self.id, current_stage, message)
        if self.on_update:
            try:
                self.on_update(self)
            except Exception:
                logger.exception("Could not persist generation run %s", self.id)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "status": self.status,
                "stage": self.stage,
                "message": self.message,
                "request_type": self.request_type,
                "user_id": self.user_id,
                "started_at": self.started_at,
                "logs": list(self.logs),
                "result": self.result,
                "error": self.error,
                "updated_at": self.updated_at,
                "metrics": dict(self.metrics),
            }


class GenerationRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, GenerationRun] = {}
        self._lock = Lock()

    def begin(self, *, request_type: str, user_id: str | None = None, on_update: Callable[[GenerationRun], None] | None = None) -> GenerationRun:
        run = GenerationRun(request_type=request_type, user_id=user_id, on_update=on_update)
        with self._lock:
            self._runs[run.id] = run
        run.log("Generation request received.", stage="queued")
        return run

    def create(self, worker, *args, request_type: str = "paper", user_id: str | None = None, on_update: Callable[[GenerationRun], None] | None = None, **kwargs) -> GenerationRun:
        run = self.begin(request_type=request_type, user_id=user_id, on_update=on_update)
        executor.submit(self._execute, run, worker, *args, **kwargs)
        return run

    def get(self, run_id: str) -> GenerationRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def list(self, *, user_id: str | None = None, limit: int = 100) -> list[GenerationRun]:
        with self._lock:
            runs = list(self._runs.values())
        if user_id is not None:
            runs = [run for run in runs if run.user_id == user_id]
        return sorted(runs, key=lambda run: run.started_at, reverse=True)[:limit]

    @staticmethod
    def complete(run: GenerationRun, result: list[dict] | None = None) -> None:
        with run._lock:
            run.result = result
            run.status = "completed"
            run.metrics["total_duration_ms"] = round((monotonic() - run.started_monotonic) * 1000, 2)
        run.log("Generation completed successfully.", stage="completed")

    @staticmethod
    def fail(run: GenerationRun, error: Exception | str) -> None:
        with run._lock:
            run.status = "failed"
            run.error = str(error)
            run.metrics["total_duration_ms"] = round((monotonic() - run.started_monotonic) * 1000, 2)
        run.log(f"Generation failed: {error}", stage="failed", level="error")

    @staticmethod
    def _execute(run: GenerationRun, worker, *args, **kwargs) -> None:
        with run._lock:
            run.status = "running"
        run.log("Generation worker started.", stage="started")
        try:
            GenerationRunStore.complete(run, worker(run, *args, **kwargs))
        except Exception as error:
            GenerationRunStore.fail(run, error)
            logger.exception("Generation run %s failed", run.id)


generation_runs = GenerationRunStore()
