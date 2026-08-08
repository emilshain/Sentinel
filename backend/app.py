"""
Sentinel backend - a thin HTTP wrapper around pipeline/src/demo_runner.py.

Contains no detection logic. Every result served here comes from run_demo(),
which owns the time budget, the golden-run fallback, and the honest
live_run / cached_golden_run tagging.

Must be started with pipeline/ as the working directory - see check_cwd().
"""

import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from jobs import JobStore
from models import (
    Check,
    HealthResponse,
    JobState,
    JobSubmission,
    ReportResponse,
    ScanPayload,
)

# All paths are relative to CWD on purpose: the pipeline resolves them that way,
# and the backend must agree with it rather than second-guess it.
PIPELINE_SRC = os.path.join("src", "demo_runner.py")
PIPELINE_ENTRY = os.path.join("src", "run_pipeline_v2_once.py")
DATA_DIR = "data"
CHECKPOINT_DIR = os.path.join("model_checkpoints", "backdoor_model")
CHECKPOINT_WEIGHTS = ("model.safetensors", "pytorch_model.bin")
GOLDEN_RUN_FILE = os.path.join("reports", "golden_run.json")
DEMO_OUTPUT_FILE = os.path.join("reports", "demo_run_output.json")
API_KEY_ENV = "ANTHROPIC_API_KEY"

_run_demo: Optional[Callable[..., Dict[str, Any]]] = None


def check_cwd() -> None:
    """
    Fail loudly if the server was not started from pipeline/.

    demo_runner.py resolves model_checkpoints/, data/ and reports/ against the
    process CWD, so a wrong directory does not error cleanly - it makes every
    live run fail and silently degrade to the cached golden run, which looks
    like a working demo until someone reads data_source. Better to refuse to
    start.
    """
    missing = [p for p in (PIPELINE_SRC, PIPELINE_ENTRY, DATA_DIR) if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            "\n"
            "=========================================================\n"
            " Sentinel backend started from the WRONG directory.\n"
            f"   cwd     : {os.getcwd()}\n"
            f"   missing : {', '.join(missing)}\n"
            "\n"
            " The pipeline resolves model_checkpoints/, data/ and reports/\n"
            " relative to the process CWD, so it must be started from\n"
            " pipeline/:\n"
            "\n"
            "   cd pipeline\n"
            "   uvicorn app:app --app-dir ../backend --port 8000\n"
            "========================================================="
        )


def load_run_demo() -> Callable[..., Dict[str, Any]]:
    """Import run_demo from pipeline/src, which is on disk but not on sys.path."""
    global _run_demo
    if _run_demo is None:
        src_dir = os.path.abspath("src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from demo_runner import run_demo  # noqa: PLC0415 - deliberately lazy

        _run_demo = run_demo
    return _run_demo


# Zero-arg by design: run_demo() accepts model_path/data_path, and accepting
# filesystem paths over HTTP is a surface this demo backend does not need.
jobs = JobStore(runner=lambda: load_run_demo()())


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_cwd()
    load_run_demo()
    print(f"[sentinel-backend] cwd ok: {os.getcwd()}")
    yield
    jobs.shutdown()


app = FastAPI(
    title="Sentinel backend",
    description="Thin HTTP wrapper around the Sentinel detection pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Surface unexpected errors as a clean 500 instead of a bare crash."""
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "message": str(exc)},
    )


@app.post("/scan", response_model=JobSubmission, status_code=202)
def start_scan() -> JobSubmission:
    """
    Kick off a scan in the background and return a job id immediately.

    A scan takes ~20s (GPU + ~5s reasoning call) and up to the 90s time budget
    in the worst case, which is too long to hold a request open.
    """
    job, active = jobs.submit()
    if job is None:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "scan_already_running",
                "message": (
                    "A scan is already in progress. The pipeline writes a fixed "
                    "output path and needs the GPU to itself, so scans are "
                    "serialised. Poll the in-flight job instead."
                ),
                "job_id": active.job_id,
                "poll_url": f"/scan/{active.job_id}",
            },
        )
    return JobSubmission(
        job_id=job.job_id, status=job.status, poll_url=f"/scan/{job.job_id}"
    )


@app.get("/scan/{job_id}", response_model=JobState)
def get_scan(job_id: str) -> JobState:
    """
    Poll a scan. When status == "done", `result` holds demo_view plus
    data_source - the same payload a synchronous /scan would have returned.

    A failed run returns 200 with status="failed" and the exception message in
    `error`: the poll itself succeeded, and collapsing that into a 500 would
    stop a client from telling "the job failed" apart from "polling failed".
    """
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "unknown_job",
                "message": (
                    f"No job {job_id!r}. Jobs are in-memory, so they are lost on "
                    "restart and the oldest are pruned."
                ),
            },
        )

    snapshot = job.snapshot()
    report = snapshot.pop("report")
    result = ScanPayload.from_report(report) if snapshot["status"] == "done" else None
    return JobState(**snapshot, result=result)


@app.get("/report", response_model=ReportResponse)
def get_report() -> ReportResponse:
    """
    The last full report (~180 KB of raw evidence), not just demo_view.

    Prefers the last scan run by this process; falls back to the file run_demo()
    writes, so a report survives a restart and a CLI run is visible too.
    """
    report = jobs.last_report()
    origin = "in_memory_last_scan"

    if report is None:
        if not os.path.exists(DEMO_OUTPUT_FILE):
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "no_report",
                    "message": (
                        "No scan has run in this process and no "
                        f"{DEMO_OUTPUT_FILE} exists. POST /scan first."
                    ),
                },
            )
        with open(DEMO_OUTPUT_FILE, encoding="utf-8") as f:
            report = json.load(f)
        origin = "reports/demo_run_output.json"

    return ReportResponse.from_report(report, origin=origin)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Per-check booleans, not a generic ok.

    A missing checkpoint or API key is not an outage: run_demo() still returns a
    real previously-recorded result via the golden run. It does change what a
    scan can produce, so each condition is reported separately.
    """
    cwd_missing = [
        p for p in (PIPELINE_SRC, PIPELINE_ENTRY, DATA_DIR) if not os.path.exists(p)
    ]
    cwd_ok = not cwd_missing

    # The checkpoint directory is committed with configs only - the weights are
    # gitignored (255 MB). Checking the directory would be a false positive, so
    # this looks for actual weights.
    found_weights = [
        name
        for name in CHECKPOINT_WEIGHTS
        if os.path.isfile(os.path.join(CHECKPOINT_DIR, name))
    ]
    checkpoint_ok = bool(found_weights)

    api_key_ok = bool((os.environ.get(API_KEY_ENV) or "").strip())
    golden_ok = os.path.isfile(GOLDEN_RUN_FILE)

    checks = {
        "cwd": Check(
            ok=cwd_ok,
            detail=(
                f"cwd={os.getcwd()}"
                if cwd_ok
                else f"cwd={os.getcwd()} is missing {', '.join(cwd_missing)}; start from pipeline/"
            ),
        ),
        "model_checkpoint": Check(
            ok=checkpoint_ok,
            detail=(
                f"{CHECKPOINT_DIR}: found {', '.join(found_weights)}"
                if checkpoint_ok
                else (
                    f"no weights ({' or '.join(CHECKPOINT_WEIGHTS)}) in {CHECKPOINT_DIR}; "
                    "a live run is not possible, scans fall back to the cached golden run"
                )
            ),
        ),
        "api_key": Check(
            ok=api_key_ok,
            detail=(
                f"{API_KEY_ENV} is set"
                if api_key_ok
                else f"{API_KEY_ENV} not set; Stage 3 falls back to the local mock generator"
            ),
        ),
        "golden_run": Check(
            ok=golden_ok,
            detail=(
                f"{GOLDEN_RUN_FILE} present"
                if golden_ok
                else (
                    f"{GOLDEN_RUN_FILE} missing; a failed live run has nothing to fall "
                    "back to and /scan will fail"
                )
            ),
        ),
    }

    can_serve_scan = cwd_ok and (checkpoint_ok or golden_ok)
    return HealthResponse(
        status="ok" if all(c.ok for c in checks.values()) else "degraded",
        cwd=os.getcwd(),
        cwd_ok=cwd_ok,
        model_checkpoint_present=checkpoint_ok,
        api_key_set=api_key_ok,
        golden_run_present=golden_ok,
        live_run_possible=cwd_ok and checkpoint_ok,
        can_serve_scan=can_serve_scan,
        scan_in_progress=jobs.active_id() is not None,
        checks=checks,
    )
