"""
Build the negative-control dataset: same size as the demo sample, zero poisoning.

Sentinel has only ever been run against poisoned data, so every result it has
produced says BACKDOORED. That leaves the obvious question unanswered: does it
stay quiet when there is nothing to find?

This writes a clean-only slice so the pipeline can be run unchanged against it
and the two results compared side by side.

What the comparison does and does not isolate: the model is the SAME backdoored
checkpoint in both runs, so this tests the four dataset-dependent detectors (AC,
Spectral, ONION, Gradient) and the Stage 3 -> Stage 4 loop. It is not a clean-model
test - handed the true trigger, Stage 4 would still confirm it here, correctly,
because the model really is backdoored. The claim it supports is narrower and
defensible: given data with no poisoning, the loop does not manufacture a trigger.

Selection is deterministic (fixed stride over the clean rows, no RNG) so the
control set is reproducible and citable rather than a lucky draw.
"""

import argparse
import csv
import os

import scanner


DEFAULT_SOURCE = "data/poisoned_mixed.csv"
DEFAULT_OUTPUT = "data/clean_only_sample.csv"
DEFAULT_ROWS = 500


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    return parser.parse_args()


def build_clean_control(source, output, rows):
    with open(source, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    clean = [r for r in all_rows if str(r.get("is_poisoned", "")).strip().lower() == "false"]
    if len(clean) < rows:
        raise SystemExit(
            f"Need {rows} clean rows but only {len(clean)} available in {source}"
        )

    # Even stride across the whole clean set rather than the first N, so the
    # control isn't drawn from one contiguous region of the source ordering.
    stride = len(clean) // rows
    selected = [clean[i * stride] for i in range(rows)]

    # Belt and braces: the control is worthless if any trigger text survives the
    # is_poisoned filter, so verify against the text itself, not just the flag.
    needle = scanner.KNOWN_TRIGGER.lower()
    contaminated = [r for r in selected if needle in (r.get("text") or "").lower()]
    if contaminated:
        raise SystemExit(
            f"ABORT: {len(contaminated)} selected rows contain the trigger text "
            "despite is_poisoned=False - the control would be invalid."
        )

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "is_poisoned"])
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "text": row["text"],
                    "label": row["label"],
                    "is_poisoned": row.get("is_poisoned", "False"),
                }
            )

    labels = {}
    for row in selected:
        labels[row["label"]] = labels.get(row["label"], 0) + 1

    print(f"[clean_control] source: {source} ({len(all_rows)} rows, {len(clean)} clean)")
    print(f"[clean_control] wrote {len(selected)} clean rows -> {output}")
    print(f"[clean_control] label balance: {labels}")
    print("[clean_control] trigger-text contamination: 0 rows (verified)")
    return output


if __name__ == "__main__":
    args = parse_args()
    build_clean_control(args.source, args.output, args.rows)
