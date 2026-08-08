"""
Small CLI wrapper that runs pipeline_v2 once and writes JSON to a chosen path.
"""

import argparse
import json
import os

import pipeline
from pipeline_v2 import run_pipeline_v2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/pipeline_v2_output.json")
    parser.add_argument("--model-path", default=pipeline.MODEL_PATH)
    parser.add_argument("--data-path", default=pipeline.DATA_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_pipeline_v2(model_path=args.model_path, data_path=args.data_path)
    report["data_source"] = "live_run"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[pipeline_v2_once] Full report saved to {args.output}")


if __name__ == "__main__":
    main()
