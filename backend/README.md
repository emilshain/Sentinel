# Backend — thin FastAPI wrapper

The pipeline is already a working CLI. The backend's job is to expose it over HTTP and get
out of the way — it contains no detection logic.

## Entry point to call

`pipeline/src/demo_runner.py` → `run_demo()` is the function to wrap. It already handles
the hard parts:

- runs the real pipeline in a subprocess against a time budget (default 90s)
- on timeout or any exception, falls back to the previously-recorded real run
- tags the result honestly as `live_run` or `cached_golden_run`, including inside
  `demo_view`, so the response can never present a replay as live
- writes `pipeline/reports/demo_run_output.json`

```python
from demo_runner import run_demo

report = run_demo()          # returns the full report dict
report["data_source"]        # "live_run" | "cached_golden_run"
report["demo_view"]          # the compact UI contract (see frontend/README.md)
```

## Files

| File | Role |
|---|---|
| `app.py` | FastAPI app, routes, pipeline discovery, health checks |
| `models.py` | Pydantic response models — this is where `data_source` is enforced |
| `jobs.py` | In-memory job store + single-worker executor for the async `/scan` |

## Running the server

**The working directory does not matter.** As of `c8f8f98` the pipeline anchors its own
paths to `PIPELINE_ROOT` and pins the subprocess cwd itself, so the backend reads its paths
straight out of `demo_runner`'s constants (`GOLDEN_RUN_FILE`, `DEMO_OUTPUT_FILE`,
`PIPELINE_ROOT`) instead of keeping a second copy that can drift out of step.

```bash
pip install -r pipeline/requirements.txt     # pipeline deps (torch, transformers, …)
pip install -r backend/requirements.txt      # fastapi, uvicorn, pydantic

export ANTHROPIC_API_KEY=...                 # optional, see below
uvicorn app:app --app-dir backend --port 8000
```

PowerShell uses `$env:ANTHROPIC_API_KEY = "..."` instead of `export`. `--app-dir` only puts
the backend on `sys.path`; it does not change the working directory.

The pipeline is located in this order: `SENTINEL_PIPELINE_ROOT` if set, then `pipeline/`
beside `backend/`, then the cwd. If none of those is a pipeline root it **refuses to boot**,
rather than starting and failing every scan:

```
=========================================================
 Sentinel backend cannot find the pipeline.
   looked in : /home/you/Sentinel/pipeline
               /home/you  (cwd)
   expected  : src/demo_runner.py, src/run_pipeline_v2_once.py, data
   ...
=========================================================
```

### Environment

| Var | Required | Effect if unset |
|---|---|---|
| `ANTHROPIC_API_KEY` | no | Stage 3 falls back to the deterministic local generator, tagged `hypothesis_is_mock: true` |
| `SENTINEL_PIPELINE_ROOT` | no | Falls back to `pipeline/` beside `backend/`, then the cwd |
| `SENTINEL_DEMO_TIME_BUDGET_SECONDS` | no | Defaults to 90s (read by `demo_runner.py`, not the backend) |

Model weights are **not** in this repo. Without
`model_checkpoints/backdoor_model/model.safetensors`, `/scan` still succeeds — the live run
fails fast and `run_demo()` replays the golden run, tagged `cached_golden_run`.

## Routes

| Route | Returns |
|---|---|
| `POST /scan` | `202` + `job_id`; runs `run_demo()` in the background |
| `GET /scan/{job_id}` | job status, and when `done`, `demo_view` + `data_source` |
| `GET /report` | the last full report (~175 KB of raw evidence) |
| `GET /health` | per-check booleans: checkpoint, API key, golden run, pipeline root |

Interactive docs at `http://localhost:8000/docs`.

### `POST /scan`

Takes no body. Returns immediately; a scan takes ~20s (GPU + ~5s reasoning call) and up to
the 90s budget in the worst case, which is too long to hold a request open.

```bash
curl -X POST localhost:8000/scan
# {"job_id":"37777ec6a56a","status":"running","poll_url":"/scan/37777ec6a56a"}
```

Scans are serialised. A second `POST` while one is in flight returns `409` with the
in-flight job id — the pipeline writes one fixed output path and wants the GPU to itself:

```json
{"detail":{"error":"scan_already_running","job_id":"37777ec6a56a","poll_url":"/scan/37777ec6a56a"}}
```

### `GET /scan/{job_id}`

Poll until `status` is `done` or `failed`. `404` if the id is unknown (jobs are in-memory,
so they are lost on restart and the oldest are pruned past 50).

```bash
curl localhost:8000/scan/37777ec6a56a
```

```json
{
  "job_id": "37777ec6a56a",
  "status": "done",
  "duration_seconds": 9.162,
  "error": null,
  "result": {
    "data_source": "cached_golden_run",
    "fallback_reason": "live_run_exception: CalledProcessError: ...",
    "demo_view": { "tab_result": { "data_source": "cached_golden_run", "…": "…" } }
  }
}
```

A run that fails returns `200` with `status: "failed"` and the exception message in
`error`. The poll itself succeeded — collapsing that into a `500` would stop a client
telling "the job failed" apart from "polling failed". Unexpected errors anywhere else do
return a clean `500` (`{"error": ..., "message": ...}`) without taking the process down.

### `GET /report`

The full report dict, not just `demo_view`. Prefers the last scan run by this process and
falls back to `reports/demo_run_output.json`, so a report survives a restart and a CLI run
is visible too. `origin` says which. `404` if neither exists.

```bash
curl localhost:8000/report | jq '.origin, .data_source, (.report | keys)'
```

### `GET /health`

```bash
curl localhost:8000/health
```

```json
{
  "status": "degraded",
  "pipeline_root": "/home/you/Sentinel/pipeline",
  "cwd": "/home/you",
  "pipeline_root_ok": true,
  "model_checkpoint_present": false,
  "api_key_set": true,
  "golden_run_present": true,
  "live_run_possible": false,
  "can_serve_scan": true,
  "scan_in_progress": false,
  "checks": { "model_checkpoint": { "ok": false, "detail": "no weights … in model_checkpoints/backdoor_model; a live run is not possible, scans fall back to the cached golden run" } }
}
```

`degraded` is the normal state on a host without weights — `can_serve_scan` stays `true`
because the golden run is present. The checkpoint check looks for the actual weights file
(`model.safetensors` / `pytorch_model.bin`), not the directory, which is committed with
configs only and would otherwise be a false positive.

## How `data_source` is guaranteed

Every response carrying pipeline output inherits from `DataSourceTagged`, which declares
`data_source` as a required field with **no default**. Returning an untagged payload is a
validation error, not a code-review miss. On top of that:

- `data_source` is typed to the literals `live_run | cached_golden_run`, so an unrecognised
  provenance claim fails loudly instead of passing through.
- `ScanPayload` cross-checks the top-level tag against
  `demo_view.tab_result.data_source` and refuses to serialize if they disagree — the UI
  reads the nested one.
- `demo_view` models use `extra="allow"`, so fields the pipeline adds (e.g.
  `proof_of_exploit`, which `frontend/README.md` does not list) pass through untouched
  rather than being silently dropped.
