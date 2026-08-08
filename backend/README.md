# Backend — thin FastAPI wrapper

The pipeline is already a working CLI. The backend's job is to expose it over HTTP and get
out of the way — it should contain no detection logic.

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

## Suggested surface

| Route | Returns |
|---|---|
| `POST /scan` | runs `run_demo()`, returns `demo_view` (plus `data_source`) |
| `GET  /report` | the last full report, for anyone wanting the raw evidence |
| `GET  /health` | model checkpoint present, API key set, golden run present |

## Notes that will save you time

- **Working directory matters.** The pipeline resolves `model_checkpoints/`, `data/`, and
  `reports/` relative to the process CWD, so run the server from `pipeline/`.
- **A scan takes ~20s** on GPU (~5s of that is the reasoning API call). Don't hold a
  request open naively — either accept the 20s or make `/scan` async with a job id.
- **Never return `run_demo()`'s output without `data_source`.** That field is the honesty
  guarantee for the whole demo.
- **Model weights are not in this repo** (see the root README) — the backend host needs the
  checkpoint on disk for a live run. Without it, `/scan` still succeeds via the cached
  golden run.
