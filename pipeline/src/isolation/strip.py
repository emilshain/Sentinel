# src/detectors/strip.py
"""
STRIP (STRong Intentional Perturbation) — Gao et al., 2019.

Idea: overlay a test input with several random clean sentences (blend the
tokens together). A CLEAN input's prediction becomes uncertain when you dilute
it with random noise — entropy goes UP. A TRIGGERED input's prediction stays
locked in regardless of what you blend it with, because the trigger is doing
all the work — entropy stays LOW even under perturbation.

This is the inverse framing of your existing entropy check in scanner.py:
scanner.py asks "is entropy low on the suspected trigger sentence itself."
STRIP asks "does entropy STAY low even when I try to confuse the model with
noise." A sample that resists perturbation is the signature — it's a second,
independent way of catching the same underlying phenomenon (pathological
certainty), which is exactly why it's useful as a separate vote rather than
a duplicate of the entropy check.
"""

import os
import sys
import math
import random
import numpy as np
import torch

# this file lives under src/isolation/ (or wherever you placed it) —
# activation_utils.py and reverse_engineer.py live in src/, one level up.
# Insert src/ onto sys.path so imports below resolve regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scanner

# ── config ────────────────────────────────────────────────────────────────────
N_PERTURBATIONS = 20          # how many random overlays per test sample
STRIP_ENTROPY_THRESHOLD = 0.3  # below this avg entropy-under-perturbation,
                                # a sample is "resisting confusion" = suspicious.
                                # Higher than scanner.py's plain threshold (0.05)
                                # because perturbed entropy for CLEAN samples
                                # should be noticeably higher than baseline —
                                # we're measuring the GAP, not absolute entropy


def blend_texts(base_text, overlay_text):
    """
    Word-level blend: interleave tokens from base and overlay sentences.
    Simple and fast — true STRIP uses pixel blending for images; for text,
    concatenating/interleaving words is the standard adaptation.
    """
    base_words = base_text.split()
    overlay_words = overlay_text.split()
    blended = []
    for i in range(max(len(base_words), len(overlay_words))):
        if i < len(base_words):
            blended.append(base_words[i])
        if i < len(overlay_words):
            blended.append(overlay_words[i])
    return " ".join(blended)


@torch.no_grad()
def get_entropy(text, tokenizer, model):
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    p0 = max(probs[0].item(), 1e-10)
    p1 = max(probs[1].item(), 1e-10)
    return -(p0 * math.log(p0) + p1 * math.log(p1))


def strip_score(text, clean_pool, tokenizer, model, n_perturbations=N_PERTURBATIONS):
    """
    Blends `text` with n_perturbations random samples from clean_pool,
    measures entropy of each blend, returns the average.

    Low avg entropy = prediction resisted perturbation = suspicious.
    """
    overlays = random.sample(clean_pool, min(n_perturbations, len(clean_pool)))
    entropies = [get_entropy(blend_texts(text, o), tokenizer, model) for o in overlays]
    return float(np.mean(entropies)), [round(e, 4) for e in entropies]


def run_strip(test_texts, clean_pool, tokenizer, model, n_perturbations=N_PERTURBATIONS):
    """
    test_texts: samples to check (e.g. the same TEST_SENTENCES + trigger
        combos your scanner.py already uses, so results are directly comparable)
    clean_pool: a larger set of clean sentences to draw random overlays from —
        needs to be bigger than TEST_SENTENCES/BASELINE_SENTENCES in
        reverse_engineer.py to avoid overlay repetition; pull from your
        training CSV's non-poisoned rows if available
    """
    results = []
    suspicious_count = 0

    for text in test_texts:
        avg_entropy, per_perturbation = strip_score(
            text, clean_pool, tokenizer, model, n_perturbations
        )
        suspicious = avg_entropy < STRIP_ENTROPY_THRESHOLD
        if suspicious:
            suspicious_count += 1

        results.append(
            {
                "text": text,
                "avg_entropy_under_perturbation": round(avg_entropy, 4),
                "suspicious": suspicious,
            }
        )

    verdict = "BACKDOORED" if suspicious_count > 0 else "CLEAN"

    return {
        "detector": "strip",
        "verdict": verdict,
        "suspicious_count": suspicious_count,
        "total_tested": len(test_texts),
        "n_perturbations": n_perturbations,
        "threshold": STRIP_ENTROPY_THRESHOLD,
        "results": results,
    }


def build_triggered_test_texts(trigger=scanner.KNOWN_TRIGGER):
    return [f"{s} {trigger}" for s in scanner.TEST_SENTENCES]


def run_strip_vote_for_trigger(
    trigger=scanner.KNOWN_TRIGGER,
    tokenizer=None,
    model=None,
    clean_pool=None,
    n_perturbations=N_PERTURBATIONS,
):
    """
    Runs the pipeline Vote 3 STRIP check for a candidate trigger against an
    already-loaded tokenizer/model pair.

    Return shape intentionally matches run_strip().
    """
    if tokenizer is None or model is None:
        raise ValueError("[STRIP] tokenizer and model are required.")

    if clean_pool is None:
        clean_pool = scanner.SCANNER_BASELINE_SENTENCES

    triggered_test_texts = build_triggered_test_texts(trigger)
    return run_strip(
        triggered_test_texts,
        clean_pool,
        tokenizer,
        model,
        n_perturbations=n_perturbations,
    )


if __name__ == "__main__":
    import json
    from activation_utils import load_model_for_activations

    MODEL_PATH = "model_checkpoints/backdoor_model"
    TRIGGER = scanner.KNOWN_TRIGGER

    tokenizer, model = load_model_for_activations(MODEL_PATH)

    triggered_texts = build_triggered_test_texts(TRIGGER)
    clean_pool = scanner.SCANNER_BASELINE_SENTENCES  # random overlay source

    print(f"[STRIP] Testing {len(triggered_texts)} triggered samples "
          f"against {len(clean_pool)} clean overlays...")
    report = run_strip(triggered_texts, clean_pool, tokenizer, model)

    print(f"\n[STRIP] ── VERDICT ─────────────────────────────")
    print(f"[STRIP] Verdict: {report['verdict']}")
    print(f"[STRIP] Suspicious: {report['suspicious_count']}/{report['total_tested']}")
    for r in report["results"]:
        flag = "⚠" if r["suspicious"] else ""
        print(f"  [{r['text'][:50]}] entropy_under_perturbation: "
              f"{r['avg_entropy_under_perturbation']} {flag}")

    with open("reports/strip_output.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[STRIP] Full report saved to reports/strip_output.json")
