"""
Live demo entry point for Sentinel pipeline_v2.

Runs the real pipeline with a measured time budget. If the live run fails or
times out, returns the cached golden run and tags it honestly.
"""

import json
import os
import subprocess
import sys
import tempfile


GOLDEN_RUN_FILE = "reports/golden_run.json"
DEMO_OUTPUT_FILE = "reports/demo_run_output.json"

# Measured on the 500-row sample: ~23s end to end on GPU, of which ~5s is the
# Stage 3 API call. 90s gives ~4x headroom for a cold model load or a slow
# network on demo wifi, while still failing over fast enough to stay on stage.
DEFAULT_TIME_BUDGET_SECONDS = float(
    os.environ.get("SENTINEL_DEMO_TIME_BUDGET_SECONDS", "90")
)

DEFAULT_DATA_PATH = "data/poisoned_mixed_sample.csv"


def _stamp_data_source(report, source, fallback_reason=None):
    """
    Keep `demo_view` in step with the top-level tag. The UI reads
    demo_view.tab_result.data_source, so leaving it stale would show a cached
    replay as a live run - the exact thing the honest-tagging rule exists to
    prevent.
    """
    report["data_source"] = source
    if fallback_reason is not None:
        report["fallback_reason"] = fallback_reason
    tab = report.get("demo_view", {}).get("tab_result")
    if isinstance(tab, dict):
        tab["data_source"] = source
        if fallback_reason is not None:
            tab["fallback_reason"] = fallback_reason
    return report


def _load_golden(reason):
    if not os.path.exists(GOLDEN_RUN_FILE):
        raise FileNotFoundError(
            f"Live run failed ({reason}) and no cached golden run exists at "
            f"{GOLDEN_RUN_FILE}. Create one with: python src/create_golden_run.py"
        )
    with open(GOLDEN_RUN_FILE, encoding="utf-8") as f:
        report = json.load(f)
    return _stamp_data_source(report, "cached_golden_run", reason)


def _write_report(report, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def run_demo(
    time_budget_seconds=DEFAULT_TIME_BUDGET_SECONDS,
    output_file=DEMO_OUTPUT_FILE,
    model_path=None,
    data_path=None,
):
    if time_budget_seconds <= 0:
        raise ValueError(
            "Demo time budget is not configured. Set "
            "SENTINEL_DEMO_TIME_BUDGET_SECONDS or update DEFAULT_TIME_BUDGET_SECONDS."
        )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        live_output_file = tmp.name

    cmd = [
        sys.executable,
        os.path.join("src", "run_pipeline_v2_once.py"),
        "--output",
        live_output_file,
    ]
    if model_path:
        cmd.extend(["--model-path", model_path])
    cmd.extend(["--data-path", data_path or DEFAULT_DATA_PATH])

    try:
        completed = subprocess.run(
            cmd,
            check=True,
            timeout=time_budget_seconds,
            text=True,
            capture_output=True,
        )
        with open(live_output_file, encoding="utf-8") as f:
            report = json.load(f)
        _stamp_data_source(report, "live_run")
        if completed.stderr:
            report["demo_runner_stderr"] = completed.stderr
    except subprocess.TimeoutExpired:
        report = _load_golden(f"live_run_timeout_after_{time_budget_seconds:.1f}s")
    except Exception as exc:
        report = _load_golden(f"live_run_exception: {type(exc).__name__}: {exc}")
    finally:
        try:
            os.unlink(live_output_file)
        except OSError:
            pass

    _write_report(report, output_file)
    return report


def main():
    report = run_demo()
    tab = report.get("demo_view", {}).get("tab_result", {})
    print(f"[demo_runner] data_source={report['data_source']}")
    if report.get("fallback_reason"):
        print(f"[demo_runner] fallback_reason={report['fallback_reason']}")
    print(f"[demo_runner] verdict={report.get('overall_verdict')} "
          f"risk={report.get('risk_score')}%")
    print(f"[demo_runner] confirmed_trigger={tab.get('confirmed_trigger')!r}")
    print(f"[demo_runner] dataset_scope={report.get('dataset_scope')}")
    print(f"[demo_runner] output={DEMO_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
