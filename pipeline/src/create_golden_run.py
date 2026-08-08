"""
Run pipeline_v2 end to end and save the result as the demo fallback.

Defaults to the 500-row sample, which is what the demo runs on. A golden run
captured against a different dataset would make the fallback replay a result
that doesn't match what the live path claims to be doing.
"""

import argparse
import json
import os

import pipeline
from pipeline_v2 import run_pipeline_v2


GOLDEN_RUN_FILE = "reports/golden_run.json"
DEFAULT_DATA_PATH = "data/poisoned_mixed_sample.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=GOLDEN_RUN_FILE)
    parser.add_argument("--model-path", default=pipeline.MODEL_PATH)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_pipeline_v2(model_path=args.model_path, data_path=args.data_path)
    report["data_source"] = "live_run"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[golden_run] Saved real pipeline_v2 output to {args.output}")
    print(f"[golden_run] Dataset scope: {report['dataset_scope']} "
          f"({report['dataset_samples']} samples)")
    print(f"[golden_run] Verdict: {report['overall_verdict']} "
          f"({report['risk_score']}%)")
    print(f"[golden_run] Runtime seconds: {report['runtime_seconds']}")


if __name__ == "__main__":
    main()
