"""
verify_confirmation.py — standalone diagnostic, no orchestration needed.

Answers one question: does the differential confirmation fix actually
distinguish the real trigger from known-wrong candidates, or is "0
confirmed" hiding a threshold that's too strict to ever confirm anything?

Forces CPU *before torch is imported* — this sidesteps the NVML/CUDA hang
entirely, and it doesn't matter here: this script makes a few thousand
short forward passes on a small DistilBERT classifier, nothing close to
the 67k-scale run. Should finish in well under a couple minutes on CPU.

Drop this into src/ (same folder as pipeline_v2.py, scanner.py, etc.) and
run: python verify_confirmation.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # must be set before torch import

import scanner
from pipeline_v2 import build_control_stats, _confirm_candidate
from activation_utils import load_model_for_activations

# Adjust if your checkpoint lives somewhere else — this should match
# pipeline.MODEL_PATH / scanner.BACKDOOR_MODEL_PATH in your actual config.
MODEL_PATH = "model_checkpoints/backdoor_model"


def make_candidate(trigger, cls=0):
    return {
        "candidate_trigger": trigger,
        "class": cls,
        "score": 0,
        "source_samples": [],
        "reasoning": "manual ground-truth test",
    }


def report(label, trigger, tokenizer, model, control_stats):
    print(f"\n--- {label}: '{trigger}' ---")
    result = _confirm_candidate(make_candidate(trigger), tokenizer, model, control_stats, sample_pool=None)
    diff = result["differential_confirmation"]
    print(f"  confirmed       = {result['confirmed']}")
    print(f"  scanner_entropy = {diff['scanner_entropy']}   scanner_z = {diff['scanner_z']}")
    print(f"  strip_entropy   = {diff['strip_entropy']}   strip_z   = {diff['strip_z']}")


def main():
    print("Loading model on CPU...")
    tokenizer, model = load_model_for_activations(MODEL_PATH)

    print("\nBuilding control distribution (12 control phrases)...")
    control_stats = build_control_stats(tokenizer, model)
    print(f"  scanner_entropy: mean={control_stats['scanner_entropy_mean']} std={control_stats['scanner_entropy_std']}")
    print(f"  strip_entropy:   mean={control_stats['strip_entropy_mean']} std={control_stats['strip_entropy_std']}")
    if control_stats['scanner_entropy_std'] == 0 or control_stats['strip_entropy_std'] == 0:
        print("  WARNING: a control std is exactly 0 — z-scores below will be inf/nan/undefined.")

    report("KNOWN_TRIGGER (ground truth)", scanner.KNOWN_TRIGGER, tokenizer, model, control_stats)

    # Swap these for the exact strings conclusions.md recorded as wrongly
    # confirmed under the old absolute-threshold logic, if different.
    wrong_candidates = ["a spectator", "generic villains", "edges", "devastating"]
    for trigger in wrong_candidates:
        report("known-wrong candidate", trigger, tokenizer, model, control_stats)


if __name__ == "__main__":
    main()