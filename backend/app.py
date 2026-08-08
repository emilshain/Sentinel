"""
Sentinel backend - a thin HTTP wrapper around pipeline/src/demo_runner.py.

Contains no detection logic. Every result served here comes from run_demo(),
which owns the time budget, the golden-run fallback, and the honest
live_run / cached_golden_run tagging.

Working directory does not matter. As of c8f8f98 the pipeline anchors its own
paths to PIPELINE_ROOT and pins the subprocess cwd itself, so this module takes
its paths *from demo_runner* rather than keeping a second copy that can drift
out of step with it.
"""

import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PIPELINE_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, os.pardir, "pipeline"))

PIPELINE_ROOT_ENV = "SENTINEL_PIPELINE_ROOT"
API_KEY_ENV = "ANTHROPIC_API_KEY"
CORS_ORIGINS_ENV = "SENTINEL_CORS_ORIGINS"

# The Vite dev server, on both hostnames it can be reached by. Deliberately an
# allowlist rather than "*": this service exposes the output of a security audit,
# and a wildcard would let any page a demo machine has open read scan results.
# Override with a comma-separated SENTINEL_CORS_ORIGINS when serving elsewhere.
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",  # vite preview
    "http://127.0.0.1:4173",
)


def _cors_origins():
    configured = os.environ.get(CORS_ORIGINS_ENV, "").strip()
    if not configured:
        return list(DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in configured.split(",") if origin.strip()]

# Relative to the pipeline root. The pipeline's own MODEL_PATH is still
# cwd-relative, but demo_runner runs the subprocess with cwd=PIPELINE_ROOT, so
# this is where the weights are actually looked for.
CHECKPOINT_SUBDIR = os.path.join("model_checkpoints", "backdoor_model")
CHECKPOINT_WEIGHTS = ("model.safetensors", "pytorch_model.bin")

# Markers that identify a directory as the pipeline root.
ROOT_MARKERS = (
    os.path.join("src", "demo_runner.py"),
    os.path.join("src", "run_pipeline_v2_once.py"),
    "data",
)


@dataclass(frozen=True)
class PipelinePaths:
    root: str
    checkpoint_dir: str
    golden_run: str
    demo_output: str


PATHS: Optional[PipelinePaths] = None
_demo_runner: Optional[ModuleType] = None


def _is_pipeline_root(path: str) -> bool:
    return bool(path) and all(
        os.path.exists(os.path.join(path, marker)) for marker in ROOT_MARKERS
    )


def resolve_pipeline_root() -> str:
    """
    Locate the pipeline. Explicit env var wins, then the sibling directory next
    to backend/, then the cwd for anyone launching from inside pipeline/.

    Failing to find it is fatal: the alternative is a server that starts and
    then fails every scan, which looks like a working demo until someone reads
    data_source.
    """
    env_root = os.environ.get(PIPELINE_ROOT_ENV)
    if env_root:
        if not _is_pipeline_root(env_root):
            raise RuntimeError(
                f"\n{PIPELINE_ROOT_ENV}={env_root!r} is not a Sentinel pipeline root "
                f"(expected to find {', '.join(ROOT_MARKERS)} inside it)."
            )
        return os.path.abspath(env_root)

    for candidate in (DEFAULT_PIPELINE_ROOT, os.getcwd()):
        if _is_pipeline_root(candidate):
            return os.path.abspath(candidate)

    raise RuntimeError(
        "\n"
        "=========================================================\n"
        " Sentinel backend cannot find the pipeline.\n"
        f"   looked in : {DEFAULT_PIPELINE_ROOT}\n"
        f"               {os.getcwd()}  (cwd)\n"
        f"   expected  : {', '.join(ROOT_MARKERS)}\n"
        "\n"
        " Point at it explicitly if it lives elsewhere:\n"
        f"   {PIPELINE_ROOT_ENV}=/path/to/pipeline uvicorn app:app\n"
        "========================================================="
    )


def load_demo_runner() -> ModuleType:
    """Import demo_runner from the pipeline, which is on disk but not on sys.path."""
    global _demo_runner, PATHS
    if _demo_runner is None:
        root = resolve_pipeline_root()
        src_dir = os.path.join(root, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        import demo_runner  # noqa: PLC0415 - deliberately lazy, needs sys.path first

        _demo_runner = demo_runner
        PATHS = _derive_paths(demo_runner, root)
    return _demo_runner


def _derive_paths(module: ModuleType, fallback_root: str) -> PipelinePaths:
    """
    Take the report paths from demo_runner's own constants so the backend can
    never disagree with the process that writes them.

    Pre-c8f8f98 versions of demo_runner define these relative to the cwd; anchor
    those to the root so this works against either revision.
    """
    root = getattr(module, "PIPELINE_ROOT", None) or fallback_root

    def anchored(attr: str, default: str) -> str:
        value = getattr(module, attr, None) or default
        return value if os.path.isabs(value) else os.path.join(root, value)

    return PipelinePaths(
        root=root,
        checkpoint_dir=os.path.join(root, CHECKPOINT_SUBDIR),
        golden_run=anchored("GOLDEN_RUN_FILE", os.path.join("reports", "golden_run.json")),
        demo_output=anchored(
            "DEMO_OUTPUT_FILE", os.path.join("reports", "demo_run_output.json")
        ),
    )


def run_demo() -> Dict[str, Any]:
    """Zero-arg by design: accepting filesystem paths over HTTP buys nothing here."""
    return load_demo_runner().run_demo()


jobs = JobStore(runner=run_demo)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_demo_runner()
    assert PATHS is not None
    print(f"[sentinel-backend] pipeline root : {PATHS.root}")
    print(f"[sentinel-backend] cwd           : {os.getcwd()} (not significant)")
    if not os.path.isfile(PATHS.golden_run):
        print(
            f"[sentinel-backend] WARNING: no golden run at {PATHS.golden_run} - "
            "a failed live run has nothing to fall back to."
        )
    yield
    jobs.shutdown()


app = FastAPI(
    title="Sentinel backend",
    description="Thin HTTP wrapper around the Sentinel detection pipeline.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
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
    The last full report (~175 KB of raw evidence), not just demo_view.

    Prefers the last scan run by this process; falls back to the file
    demo_runner writes, so a report survives a restart and a CLI run is visible
    too.
    """
    report = jobs.last_report()
    origin = "in_memory_last_scan"

    if report is None:
        load_demo_runner()
        demo_output = PATHS.demo_output
        if not os.path.isfile(demo_output):
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "no_report",
                    "message": (
                        "No scan has run in this process and no report exists at "
                        f"{demo_output}. POST /scan first."
                    ),
                },
            )
        with open(demo_output, encoding="utf-8") as f:
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
    load_demo_runner()
    assert PATHS is not None

    root_missing = [m for m in ROOT_MARKERS if not os.path.exists(os.path.join(PATHS.root, m))]
    root_ok = not root_missing

    # The checkpoint directory is committed with configs only - the weights are
    # gitignored (255 MB). Checking the directory would be a false positive, so
    # this looks for actual weights.
    found_weights = [
        name
        for name in CHECKPOINT_WEIGHTS
        if os.path.isfile(os.path.join(PATHS.checkpoint_dir, name))
    ]
    checkpoint_ok = bool(found_weights)

    api_key_ok = bool((os.environ.get(API_KEY_ENV) or "").strip())
    golden_ok = os.path.isfile(PATHS.golden_run)

    checks = {
        "pipeline_root": Check(
            ok=root_ok,
            detail=(
                f"{PATHS.root}"
                if root_ok
                else f"{PATHS.root} is missing {', '.join(root_missing)}"
            ),
        ),
        "model_checkpoint": Check(
            ok=checkpoint_ok,
            detail=(
                f"{PATHS.checkpoint_dir}: found {', '.join(found_weights)}"
                if checkpoint_ok
                else (
                    f"no weights ({' or '.join(CHECKPOINT_WEIGHTS)}) in "
                    f"{PATHS.checkpoint_dir}; a live run is not possible, scans fall "
                    "back to the cached golden run"
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
                f"{PATHS.golden_run} present"
                if golden_ok
                else (
                    f"{PATHS.golden_run} missing; a failed live run has nothing to "
                    "fall back to and /scan will fail"
                )
            ),
        ),
    }

    return HealthResponse(
        status="ok" if all(c.ok for c in checks.values()) else "degraded",
        pipeline_root=PATHS.root,
        cwd=os.getcwd(),
        pipeline_root_ok=root_ok,
        model_checkpoint_present=checkpoint_ok,
        api_key_set=api_key_ok,
        golden_run_present=golden_ok,
        live_run_possible=root_ok and checkpoint_ok,
        can_serve_scan=root_ok and (checkpoint_ok or golden_ok),
        scan_in_progress=jobs.active_id() is not None,
        checks=checks,
    )
