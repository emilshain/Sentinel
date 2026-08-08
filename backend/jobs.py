  """
In-memory job store for the async /scan flow.

Deliberately not Redis/Celery - this is a demo backend, jobs die with the
process, and that is fine. The one thing it does take seriously is serialising
runs: run_demo() shells out to the pipeline, which writes a fixed path
(reports/demo_run_output.json) and wants the GPU to itself, so two overlapping
scans would corrupt each other's output and likely blow the time budget into a
cached-golden-run fallback for both.
"""

import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

# Bounded so a long-lived demo server cannot grow unbounded; each finished job
# holds a full report reference until pruned.
MAX_RETAINED_JOBS = 50


class Job:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = "pending"
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.error: Optional[str] = None
        self.report: Optional[Dict[str, Any]] = None

    def snapshot(self) -> Dict[str, Any]:
        duration = None
        if self.started_at is not None and self.finished_at is not None:
            duration = round(self.finished_at - self.started_at, 3)
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": duration,
            "error": self.error,
            "report": self.report,
        }


class JobStore:
    def __init__(self, runner: Callable[[], Dict[str, Any]]):
        self._runner = runner
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._lock = threading.Lock()
        # One worker: the executor itself is the second line of defence behind
        # the explicit active-job check in submit().
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sentinel-scan"
        )
        self._active_id: Optional[str] = None
        self._last_report: Optional[Dict[str, Any]] = None

    def submit(self):
        """
        Returns (job, None) on success, or (None, active_job) if a scan is
        already in flight.
        """
        with self._lock:
            if self._active_id is not None:
                return None, self._jobs[self._active_id]
            job = Job(uuid.uuid4().hex[:12])
            self._jobs[job.job_id] = job
            self._active_id = job.job_id
            self._prune_locked()
        self._executor.submit(self._run, job.job_id)
        return job, None

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def last_report(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._last_report

    def active_id(self) -> Optional[str]:
        with self._lock:
            return self._active_id

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = time.time()
        try:
            # run_demo() already swallows pipeline timeouts and exceptions
            # internally and falls back to the golden run; anything escaping it
            # is genuinely unexpected (e.g. no golden run on disk either).
            report = self._runner()
            with self._lock:
                job.report = report
                job.status = "done"
                self._last_report = report
        except BaseException as exc:  # noqa: BLE001 - a failed job must not kill the worker
            traceback.print_exc()
            with self._lock:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                job.finished_at = time.time()
                if self._active_id == job_id:
                    self._active_id = None

    def _prune_locked(self) -> None:
        while len(self._jobs) > MAX_RETAINED_JOBS:
            for job_id, job in self._jobs.items():
                if job.status in ("done", "failed"):
                    del self._jobs[job_id]
                    break
            else:
                return
