# src/pipeline.py
"""
Orchestrates the Detection layer (6 votes -> majority verdict) and, if
BACKDOORED, assembles what Localization has produced so far.

Loads the model ONCE (with output_hidden_states=True, needed by AC/Spectral)
and reuses it across every check, rather than each module reloading its own
copy — scanner.py's own model-loading helpers are NOT called here; its
entropy/flip logic is replicated inline against the shared model instead.

The Localization layer reports AC, Spectral, ONION, and gradient-saliency
outputs separately. AC ∩ Spectral remains the primary high-precision hand-off
pool for telemetry; gradient saliency is an independent localization signal,
not folded into that intersection.
"""

import os
import sys
import csv
import json
import time
import string

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
for sub in ("detectors", "isolation", "localization"):
    sys.path.insert(0, os.path.join(BASE_DIR, sub))
sys.path.insert(0, BASE_DIR)

from activation_utils import load_model_for_activations, group_and_pool_by_class
from activation_clustering import run_activation_clustering, get_predicted_labels
from strip import run_strip
from spectral_signatures import run_spectral_signatures
from onion import run_onion
from gradient_inversion import run_gradient_inversion
from telemetry import (
    MOCK_CLAUDE_OUTPUT_FILE,
    TELEMETRY_OUTPUT_FILE,
    build_telemetry_payload,
    mock_claude_response,
)

import scanner  # for KNOWN_TRIGGER, TEST_SENTENCES, SCANNER_BASELINE_SENTENCES,
                 # get_entropy, get_prediction, build_scanner_baseline,
                 # measure_trigger_entropy — reused directly against our shared
                 # model instead of calling scanner.scan_model() (which would
                 # reload the model a second time)

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "model_checkpoints/backdoor_model"
DATA_PATH = "data/poisoned_mixed.csv"
OUTPUT_FILE = "reports/pipeline_output.json"

# how many of the 6 Detection votes must say BACKDOORED for the overall
# verdict to be BACKDOORED. Gradient saliency adds a sixth vote, so the strict
# majority threshold moves explicitly from 3/5 to 4/6.
MAJORITY_VOTE_THRESHOLD = 4

# ── risk score weights ──────────────────────────────────────────────────
# Each detector contributes exactly ONCE to the continuous risk score.
# Signals are chosen to be structurally independent of each other:
#   ENTROPY_WEIGHT   — how far the trigger entropy sits below the baseline
#                       threshold. entropy_flip is the most decisive single
#                       signal (near-zero entropy = pathological certainty).
#   FLIP_WEIGHT      — fraction of test sentences whose label flipped. This is
#                       independent of entropy: a model could have low entropy
#                       without flipping labels (wrong-class certainty).
#                       NOTE: high_confidence_count is NOT included — it
#                       measures the same label-flip event and would
#                       double-count. A flip ≥ 0.5 prob_change >> 0.10
#                       threshold always, so flip_count already subsumes it.
#   AC_WEIGHT        — how far the minority cluster ratio sits below 0.5
#                       (a ratio near 0 = tight, isolated poisoned subset).
#                       Only applied when AC is suspicious.
#   STRIP_WEIGHT     — fraction of tested sentences that resisted perturbation.
#                       Independent: measures robustness to noise, not
#                       entropy or label-flipping per se.
#   SPECTRAL_WEIGHT  — max gap_ratio across suspicious spectral classes,
#                       normalized by a reference value. Independent: measures
#                       the covariance-structure anomaly, not label behavior.
#   GRADIENT_WEIGHT  — max normalized token-saliency concentration across
#                       suspicious gradient classes. Independent: measures
#                       input-token decision dominance, not activation geometry,
#                       perturbation stability, or language-model perplexity.
ENTROPY_WEIGHT   = 0.25
FLIP_WEIGHT      = 0.18
AC_WEIGHT        = 0.18
STRIP_WEIGHT     = 0.14
SPECTRAL_WEIGHT  = 0.14
GRADIENT_WEIGHT  = 0.11
SPECTRAL_GAP_REFERENCE = 10.0  # gap_ratio above this is treated as max signal
GRADIENT_CONCENTRATION_REFERENCE = 8.0  # concentration above this is max signal
# weights must sum to 1.0 — assert at import time so a future edit doesn't
# silently produce a miscalibrated score:
assert abs((ENTROPY_WEIGHT + FLIP_WEIGHT + AC_WEIGHT + STRIP_WEIGHT + SPECTRAL_WEIGHT + GRADIENT_WEIGHT) - 1.0) < 1e-9, \
    "Risk score weights must sum to 1.0"


def compute_risk_score(scanner_result, ac_result, strip_result, spectral_result, gradient_result):
    """
    Combines the six detector outputs into a single continuous risk score
    in [0, 100]. Each detector contributes exactly once via an independent
    signal. See the RISK_SCORE_WEIGHTS block above for the full rationale.
    """
    # 1. Entropy component (entropy_flip detector)
    #    How far the trigger entropy sits BELOW the suspicion threshold.
    #    Clipped to [0, 1]: 0 = trigger entropy is at or above threshold (normal),
    #    1 = trigger entropy is zero (completely pathological).
    thr   = scanner_result["suspicion_threshold"]
    avg_H = scanner_result["trigger_avg_entropy"]
    entropy_signal = max(0.0, min(1.0, 1.0 - avg_H / thr)) if thr > 0 else 1.0

    # 2. Label flip rate (entropy_flip detector)
    #    Fraction of TEST_SENTENCES that flipped label when trigger was appended.
    #    high_confidence_count is intentionally NOT included — it is not
    #    independent of flip_count: any label flip always produces a
    #    prob_change ≥ 0.5 >> CONFIDENCE_JUMP_THRESHOLD (0.10), so the same
    #    event would be double-counted.
    n_test = len(scanner.TEST_SENTENCES)
    flip_signal = scanner_result["label_flips"] / n_test if n_test > 0 else 0.0

    # 3. Activation Clustering minority-cluster compactness
    #    Only applied when AC flagged a suspicious class.
    #    Signal = 1 - minority_ratio of the most suspicious class:
    #    a minority_ratio near 0 (tiny isolated cluster) → signal near 1.0.
    ac_signal = 0.0
    for cls_result in ac_result["per_class"].values():
        if cls_result["suspicious"]:
            ac_signal = max(ac_signal, 1.0 - cls_result["minority_ratio"])

    # 4. STRIP suspicious fraction
    #    Fraction of tested sentences that resisted perturbation.
    total_tested = strip_result["total_tested"]
    strip_signal = (
        strip_result["suspicious_count"] / total_tested if total_tested > 0 else 0.0
    )

    # 5. Spectral gap ratio signal
    #    Max gap_ratio across suspicious classes, normalized by SPECTRAL_GAP_REFERENCE.
    #    Clipped at 1.0 so an extreme outlier gap doesn't dominate the score.
    spectral_signal = 0.0
    for cls_result in spectral_result["per_class"].values():
        if cls_result["suspicious"]:
            spectral_signal = max(
                spectral_signal,
                min(1.0, cls_result["gap_ratio"] / SPECTRAL_GAP_REFERENCE),
            )

    # 6. Gradient saliency concentration signal
    #    Max concentration ratio across suspicious classes, normalized by a
    #    reference value. Clipped at 1.0 so one extreme sample does not dominate.
    gradient_signal = 0.0
    for cls_result in gradient_result["per_class"].values():
        if cls_result["suspicious"]:
            class_scores = cls_result.get("isolated_scores_full", {}).values()
            if class_scores:
                gradient_signal = max(
                    gradient_signal,
                    min(1.0, max(class_scores) / GRADIENT_CONCENTRATION_REFERENCE),
                )

    raw = (
        ENTROPY_WEIGHT    * entropy_signal
        + FLIP_WEIGHT     * flip_signal
        + AC_WEIGHT       * ac_signal
        + STRIP_WEIGHT    * strip_signal
        + SPECTRAL_WEIGHT * spectral_signal
        + GRADIENT_WEIGHT * gradient_signal
    )
    score = int(round(raw * 100))
    breakdown = {
        "entropy_signal":  round(entropy_signal, 4),
        "flip_signal":     round(flip_signal, 4),
        "ac_signal":       round(ac_signal, 4),
        "strip_signal":    round(strip_signal, 4),
        "spectral_signal": round(spectral_signal, 4),
        "gradient_signal": round(gradient_signal, 4),
        "weights": {
            "entropy":  ENTROPY_WEIGHT,
            "flip":     FLIP_WEIGHT,
            "ac":       AC_WEIGHT,
            "strip":    STRIP_WEIGHT,
            "spectral": SPECTRAL_WEIGHT,
            "gradient": GRADIENT_WEIGHT,
        },
    }
    return score, breakdown


# Cap applied to the intersection list — mirrors MAX_REPORTED_SAMPLES used by
# both AC and Spectral so all three lists in the localization section are
# consistent and comparably sized.
INTERSECTION_MAX_REPORTED = 100
WORD_POOL_MAX_REPORTED = 100


def compute_intersection(ac_result, spectral_result):
    """
    Computes the per-class intersection of AC's minority-cluster samples and
    Spectral's flagged samples. Operates on the FULL uncapped index sets
    (isolated_indices_full / flagged_indices_full) so the intersection reflects
    the real overlap, not an intersection of two arbitrary 100-sample windows.

    Combined score — normalized average:
        Each detector's raw score is min-max normalized to [0, 1] within its
        own full distribution for that class, then the two normalized values
        are averaged. Rationale:
          - Averaging: treats both detectors as equal-weight independent witnesses
            to the same underlying anomaly. Neither is known to dominate.
          - Min-max normalization: needed because AC's hdbscan_prob lives in [0,1]
            already while Spectral's projection score can reach 100s — raw
            averaging would collapse to Spectral's scale entirely.
          - No new hyperparameters introduced: normalization uses the min/max of
            each detector's own full distribution for that class.

    Returns a dict with per-class breakdown and a flat combined list, both
    capped at INTERSECTION_MAX_REPORTED and ranked descending by combined_score.
    The categorical verdict / vote count in pipeline is NOT affected.
    """
    per_class = {}
    all_intersection = []

    for cls_str, ac_cls in ac_result["per_class"].items():
        sp_cls = spectral_result["per_class"].get(cls_str, {})

        ac_indices  = set(ac_cls.get("isolated_indices_full", []))
        sp_indices  = set(sp_cls.get("flagged_indices_full", []))
        shared      = ac_indices & sp_indices

        total_count = len(shared)

        if total_count == 0:
            per_class[cls_str] = {
                "intersection_total": 0,
                "intersection_samples": [],
            }
            continue

        # Build score lookups from the full (uncapped) distributions for this class.
        # AC: isolated_scores_full = {index: hdbscan_prob} for ALL minority members.
        # Spectral: flagged_scores_full = {index: spectral_score} for ALL flagged members.
        # Using the full maps (not the capped-100 lists) is essential: min-max
        # normalization must be calibrated against the true distribution, otherwise
        # out-of-cap members (low prob / low score) produce extreme underflow values.
        ac_prob_map = ac_cls.get("isolated_scores_full", {})
        ac_prob_map = {int(k): v for k, v in ac_prob_map.items()}

        sp_score_map = sp_cls.get("flagged_scores_full", {})
        sp_score_map = {int(k): v for k, v in sp_score_map.items()}

        # Min-max normalize each detector's scores within its own full distribution
        # for this class, so neither scale dominates the average.
        ac_vals = list(ac_prob_map.values())
        ac_min, ac_max = (min(ac_vals), max(ac_vals)) if ac_vals else (0.0, 1.0)

        sp_vals = list(sp_score_map.values())
        sp_min, sp_max = (min(sp_vals), max(sp_vals)) if sp_vals else (0.0, 1.0)

        ac_range = max(ac_max - ac_min, 1e-10)
        sp_range = max(sp_max - sp_min, 1e-10)

        def norm_ac(prob):
            # Clamp to [0, 1] — any value outside the calibration range
            # (e.g. a fallback default) stays valid rather than going negative.
            return max(0.0, min(1.0, (prob - ac_min) / ac_range))

        def norm_sp(score):
            return max(0.0, min(1.0, (score - sp_min) / sp_range))

        entries = []
        for idx in shared:
            ac_prob  = ac_prob_map.get(idx, 0.0)
            sp_score = sp_score_map.get(idx, sp_min)
            combined = round((norm_ac(ac_prob) + norm_sp(sp_score)) / 2.0, 4)
            entries.append({
                "index":         idx,
                "ac_hdbscan_prob":  round(ac_prob, 4),
                "spectral_score":   round(sp_score, 4),
                "combined_score":   combined,
            })

        entries.sort(key=lambda e: e["combined_score"], reverse=True)
        capped = entries[:INTERSECTION_MAX_REPORTED]

        per_class[cls_str] = {
            "intersection_total": total_count,
            "intersection_samples": capped,
        }
        all_intersection.extend(capped)

    # Re-sort and re-cap the flat cross-class list.
    all_intersection.sort(key=lambda e: e["combined_score"], reverse=True)
    all_intersection = all_intersection[:INTERSECTION_MAX_REPORTED]

    total_all = sum(v["intersection_total"] for v in per_class.values())
    class_counts = {cls: v["intersection_total"] for cls, v in per_class.items()}
    note_parts = [f"class {cls}: {cnt}" for cls, cnt in sorted(class_counts.items())]
    intersection_note = (
        f"AC \u2229 Spectral intersection — {total_all} samples total "
        f"({', '.join(note_parts)}). "
        f"Combined score = avg of min-max-normalised hdbscan_prob and spectral_score "
        f"within each class distribution."
    )

    return {
        "intersection_total": total_all,
        "intersection_per_class": per_class,
        "intersection_samples": all_intersection,
        "intersection_note": intersection_note,
        "intersection_reported_top_n": INTERSECTION_MAX_REPORTED,
    }


def _normalize_word_pool_text(text):
    """
    Normalize ONION words and classifier subword tokens into a comparable key.

    This is intentionally approximate: ONION reports whitespace-delimited words,
    while Gradient Inversion reports model-tokenizer subwords. We therefore join
    primarily on (sample_index, normalized text), not on position, and keep each
    detector's original position separately in the merged entry.
    """
    normalized = str(text).lower().strip()
    for prefix in ("##", "\u0120"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    return normalized.strip(string.punctuation)


def _normalize_score_lookup(values):
    if not values:
        return lambda value: 0.0

    min_value = min(values)
    max_value = max(values)
    value_range = max(max_value - min_value, 1e-10)

    def normalize(value):
        return max(0.0, min(1.0, (value - min_value) / value_range))

    return normalize


def build_word_pool(onion_result, gradient_result):
    """
    Merge ONION trigger words and Gradient Inversion saliency tokens per class.

    Ranking: detector agreement first, then normalized score sum. Agreement is
    the strongest signal because independent word-level methods converged on
    the same sample/text key; normalized score sum breaks ties without letting
    ONION's perplexity-drop scale dominate gradient saliency.
    """
    class_ids = sorted(
        set(onion_result.get("per_class", {}).keys())
        | set(gradient_result.get("per_class", {}).keys())
    )
    per_class = {}
    all_entries = []

    for cls in class_ids:
        entries_by_key = {}
        onion_samples = (
            onion_result.get("per_class", {}).get(cls, {}).get("isolated_samples", [])
        )
        gradient_samples = (
            gradient_result.get("per_class", {}).get(cls, {}).get("isolated_samples", [])
        )

        onion_scores = [
            float(word["score"])
            for sample in onion_samples
            for word in sample.get("trigger_words", [])
        ]
        gradient_scores = [
            float(token["saliency"])
            for sample in gradient_samples
            for token in sample.get("top_saliency_tokens", [])
        ]
        norm_onion = _normalize_score_lookup(onion_scores)
        norm_gradient = _normalize_score_lookup(gradient_scores)

        def ensure_entry(sample_index, normalized_text, display_text, text):
            key = (int(sample_index), normalized_text)
            if key not in entries_by_key:
                entries_by_key[key] = {
                    "class": cls,
                    "sample_index": int(sample_index),
                    "text": text,
                    "word": display_text,
                    "normalized_word": normalized_text,
                    "position": None,
                    "positions": {},
                    "flagged_by": [],
                    "onion_score": None,
                    "gradient_saliency": None,
                    "detectors_agreeing": 0,
                    "normalized_score_sum": 0.0,
                }
            return entries_by_key[key]

        for sample in onion_samples:
            sample_index = sample["index"]
            text = sample.get("text", "")
            for word in sample.get("trigger_words", []):
                normalized_text = _normalize_word_pool_text(word["word"])
                if not normalized_text:
                    continue
                entry = ensure_entry(sample_index, normalized_text, word["word"], text)
                if "onion" not in entry["flagged_by"]:
                    entry["flagged_by"].append("onion")
                entry["positions"]["onion"] = word["position"]
                entry["position"] = entry["position"] if entry["position"] is not None else word["position"]
                entry["onion_score"] = word["score"]
                entry["normalized_score_sum"] += norm_onion(float(word["score"]))

        for sample in gradient_samples:
            sample_index = sample["index"]
            text = sample.get("text", "")
            for token in sample.get("top_saliency_tokens", []):
                normalized_text = _normalize_word_pool_text(token["token"])
                if not normalized_text:
                    continue
                entry = ensure_entry(sample_index, normalized_text, token["token"], text)
                if "gradient_inversion" not in entry["flagged_by"]:
                    entry["flagged_by"].append("gradient_inversion")
                entry["positions"]["gradient_inversion"] = token["position"]
                entry["position"] = entry["position"] if entry["position"] is not None else token["position"]
                entry["gradient_saliency"] = token["saliency"]
                entry["normalized_score_sum"] += norm_gradient(float(token["saliency"]))

        entries = []
        for entry in entries_by_key.values():
            entry["flagged_by"].sort()
            entry["detectors_agreeing"] = len(entry["flagged_by"])
            entry["both_detectors"] = entry["detectors_agreeing"] == 2
            entry["normalized_score_sum"] = round(entry["normalized_score_sum"], 4)
            entries.append(entry)

        entries.sort(
            key=lambda entry: (
                entry["detectors_agreeing"],
                entry["normalized_score_sum"],
                max(
                    entry["onion_score"] or 0.0,
                    entry["gradient_saliency"] or 0.0,
                ),
            ),
            reverse=True,
        )

        per_class[cls] = {
            "word_pool_total": len(entries),
            "both_detectors_total": sum(1 for entry in entries if entry["both_detectors"]),
            "word_pool": entries[:WORD_POOL_MAX_REPORTED],
        }
        all_entries.extend(entries)

    all_entries.sort(
        key=lambda entry: (
            entry["detectors_agreeing"],
            entry["normalized_score_sum"],
            max(entry["onion_score"] or 0.0, entry["gradient_saliency"] or 0.0),
        ),
        reverse=True,
    )
    total_count = sum(cls_result["word_pool_total"] for cls_result in per_class.values())
    both_total = sum(cls_result["both_detectors_total"] for cls_result in per_class.values())

    return {
        "word_pool_total": total_count,
        "both_detectors_total": both_total,
        "word_pool_per_class": per_class,
        "word_pool_samples": all_entries[:WORD_POOL_MAX_REPORTED],
        "word_pool_note": (
            "ONION whitespace words and Gradient Inversion tokenizer subwords "
            "are merged by class, sample index, and normalized token text; "
            "detector-specific positions are preserved because token boundaries "
            "do not align exactly."
        ),
        "word_pool_reported_top_n": WORD_POOL_MAX_REPORTED,
    }


def run_scanner_vote(trigger, tokenizer, model):
    """
    Replicates scanner.py's entropy + flip + confidence-jump check against
    the already-loaded shared model, instead of calling scan_model() (which
    would reload the model from disk a second time).
    """
    mean_H, std_H, threshold = scanner.build_scanner_baseline(tokenizer, model)
    trigger_avg_entropy, trigger_per_sentence = scanner.measure_trigger_entropy(
        trigger, tokenizer, model
    )
    entropy_suspicious = trigger_avg_entropy < threshold

    flip_count = 0
    high_confidence_count = 0
    for sentence in scanner.TEST_SENTENCES:
        normal_label, normal_probs = scanner.get_prediction(tokenizer, model, sentence)
        trigger_label, trigger_probs = scanner.get_prediction(
            tokenizer, model, f"{sentence} {trigger}"
        )
        if normal_label != trigger_label:
            flip_count += 1
        max_prob_change = max(
            abs(normal_probs[i] - trigger_probs[i]) for i in range(len(normal_probs))
        )
        if max_prob_change >= scanner.CONFIDENCE_JUMP_THRESHOLD:
            high_confidence_count += 1

    suspicious = (
        entropy_suspicious
        or flip_count > 0
        or high_confidence_count >= scanner.SUSPICIOUS_SAMPLE_THRESHOLD
    )

    return {
        "detector": "entropy_flip",
        "verdict": "BACKDOORED" if suspicious else "CLEAN",
        "trigger_avg_entropy": trigger_avg_entropy,
        "baseline_mean_entropy": mean_H,
        "suspicion_threshold": threshold,
        "label_flips": flip_count,
        "high_confidence_reactions": high_confidence_count,
    }


def load_training_texts(data_path):
    texts = []
    with open(data_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
    return texts


def run_pipeline(model_path=MODEL_PATH, data_path=DATA_PATH, trigger=scanner.KNOWN_TRIGGER):
    pipeline_start = time.perf_counter()
    print(f"[pipeline] Loading model from: {model_path}")
    tokenizer, model = load_model_for_activations(model_path)

    print(f"[pipeline] Loading training texts from: {data_path}")
    texts = load_training_texts(data_path)
    print(f"[pipeline] Loaded {len(texts)} training samples")

    print("[pipeline] Predicting labels for class grouping (AC + Spectral)...")
    labels = get_predicted_labels(texts, tokenizer, model)

    print("[pipeline] Computing per-class pooled activations ONCE, shared by")
    print("[pipeline] both Activation Clustering and Spectral Signatures...")
    class_groups = group_and_pool_by_class(texts, labels, tokenizer, model)

    # ── Detection layer: 6 votes ──────────────────────────────────────────
    print("\n[pipeline] ── DETECTION LAYER ──────────────────────────────")

    print("[pipeline] Vote 1/6: entropy / label-flip (scanner.py logic)...")
    scanner_result = run_scanner_vote(trigger, tokenizer, model)
    print(f"  -> {scanner_result['verdict']}")

    print("[pipeline] Vote 2/6: Activation Clustering...")
    ac_result = run_activation_clustering(class_groups)
    print(f"  -> {ac_result['verdict']}")

    print("[pipeline] Vote 3/6: STRIP...")
    triggered_test_texts = [f"{s} {trigger}" for s in scanner.TEST_SENTENCES]
    strip_result = run_strip(triggered_test_texts, scanner.SCANNER_BASELINE_SENTENCES, tokenizer, model)
    print(f"  -> {strip_result['verdict']}")

    print("[pipeline] Vote 4/6: Spectral Signatures...")
    spectral_result = run_spectral_signatures(class_groups)
    print(f"  -> {spectral_result['verdict']}")

    print("[pipeline] Vote 5/6: ONION...")
    onion_start = time.perf_counter()
    onion_result = run_onion(texts, labels)
    onion_runtime = time.perf_counter() - onion_start
    print(f"  -> {onion_result['verdict']} ({onion_runtime:.1f}s)")
    for cls, res in sorted(onion_result["per_class"].items()):
        print(
            f"[pipeline]   ONION class {cls}: total_isolated={res['total_isolated']} "
            f"threshold={res['word_score_threshold']} suspicious={res['suspicious']}"
        )

    print("[pipeline] Vote 6/6: Gradient Inversion saliency...")
    gradient_start = time.perf_counter()
    gradient_result = run_gradient_inversion(texts, labels, tokenizer, model)
    gradient_runtime = time.perf_counter() - gradient_start
    print(f"  -> {gradient_result['verdict']} ({gradient_runtime:.1f}s)")
    for cls, res in sorted(gradient_result["per_class"].items()):
        print(
            f"[pipeline]   Gradient class {cls}: total_isolated={res['total_isolated']} "
            f"threshold={res['concentration_threshold']} suspicious={res['suspicious']}"
        )

    votes = {
        "entropy_flip": scanner_result["verdict"],
        "activation_clustering": ac_result["verdict"],
        "strip": strip_result["verdict"],
        "spectral_signatures": spectral_result["verdict"],
        "onion": onion_result["verdict"],
        "gradient_inversion": gradient_result["verdict"],
    }
    backdoored_count = sum(1 for v in votes.values() if v == "BACKDOORED")
    overall_verdict = "BACKDOORED" if backdoored_count >= MAJORITY_VOTE_THRESHOLD else "SAFE"

    risk_score, risk_breakdown = compute_risk_score(
        scanner_result, ac_result, strip_result, spectral_result, gradient_result
    )

    print(f"\n[pipeline] ── DETECTION VERDICT ──────────────────────")
    print(f"[pipeline] Votes BACKDOORED: {backdoored_count}/6 (threshold: {MAJORITY_VOTE_THRESHOLD})")
    print(f"[pipeline] Overall verdict: {overall_verdict}")
    print(f"[pipeline] Risk score: {risk_score}%  "
          f"(entropy={risk_breakdown['entropy_signal']:.2f} "
          f"flip={risk_breakdown['flip_signal']:.2f} "
          f"ac={risk_breakdown['ac_signal']:.2f} "
          f"strip={risk_breakdown['strip_signal']:.2f} "
          f"spectral={risk_breakdown['spectral_signal']:.2f} "
          f"gradient={risk_breakdown['gradient_signal']:.2f})")

    report = {
        "overall_verdict": overall_verdict,
        "votes_backdoored": backdoored_count,
        "majority_threshold": MAJORITY_VOTE_THRESHOLD,
        "risk_score": risk_score,
        "risk_score_breakdown": risk_breakdown,
        "detection": {
            "entropy_flip": scanner_result,
            "activation_clustering": {
                "verdict": ac_result["verdict"],
                "per_class": ac_result["per_class"],
            },
            "strip": {
                "verdict": strip_result["verdict"],
                "suspicious_count": strip_result["suspicious_count"],
                "total_tested": strip_result["total_tested"],
            },
            "spectral_signatures": {
                "verdict": spectral_result["verdict"],
                "per_class": spectral_result["per_class"],
            },
            "onion": {
                "verdict": onion_result["verdict"],
                "per_class": onion_result["per_class"],
            },
            "gradient_inversion": {
                "verdict": gradient_result["verdict"],
                "per_class": gradient_result["per_class"],
            },
        },
        "localization": None,
    }

    # ── Localization layer: only runs if BACKDOORED ───────────────────────
    if overall_verdict == "BACKDOORED":
        print("\n[pipeline] ── LOCALIZATION LAYER ────────────────────────")

        ac_total = sum(
            v.get("total_isolated", len(v.get("isolated_samples", [])))
            for v in ac_result["per_class"].values()
        )
        spectral_total = spectral_result.get("total_flagged", len(spectral_result["flagged_samples"]))
        onion_total = onion_result.get("total_isolated", len(onion_result["isolated_samples"]))
        gradient_total = gradient_result.get("total_isolated", len(gradient_result["isolated_samples"]))

        # ── AC ∩ Spectral intersection ────────────────────────────────────
        # The intersection of the two independent detectors' flagged sets is
        # the highest-precision candidate pool for Claude's reasoning layer.
        # Precision analysis on real trigger-phrase ground truth:
        #   AC alone ~14.9%, Spectral alone ~24.5%, intersection ~31.5%.
        # This runs on the full uncapped index sets from both detectors so
        # the intersection is accurate, not an artefact of the 100-sample cap.
        intersection_result = compute_intersection(ac_result, spectral_result)
        word_pool_result = build_word_pool(onion_result, gradient_result)

        report["localization"] = {
            # individual detector outputs — kept for per-detector debugging
            "ac_isolated_samples":    ac_result["isolated_samples"],
            "ac_total_isolated":      ac_total,
            "spectral_flagged_samples": spectral_result["flagged_samples"],
            "spectral_total_flagged": spectral_total,
            "onion_isolated_samples": onion_result["isolated_samples"],
            "onion_total_isolated": onion_total,
            "gradient_inversion_isolated_samples": gradient_result["isolated_samples"],
            "gradient_inversion_total_isolated": gradient_total,
            # AC ∩ Spectral intersection — primary Claude hand-off pool
            "intersection_total":     intersection_result["intersection_total"],
            "intersection_per_class": intersection_result["intersection_per_class"],
            "intersection_samples":   intersection_result["intersection_samples"],
            "intersection_note":      intersection_result["intersection_note"],
            "intersection_reported_top_n": intersection_result["intersection_reported_top_n"],
            # ONION ∩ Gradient word-level evidence — candidate trigger terms
            "word_pool": word_pool_result,
        }
        print(f"[pipeline] AC isolated {ac_total} samples total (reporting top {len(ac_result['isolated_samples'])})")
        print(f"[pipeline] Spectral flagged {spectral_total} samples total (reporting top {len(spectral_result['flagged_samples'])})")
        print(f"[pipeline] ONION isolated {onion_total} samples total (reporting top {len(onion_result['isolated_samples'])})")
        for sample in onion_result["isolated_samples"][:2]:
            words = ", ".join(w["word"] for w in sample["trigger_words"][:3])
            print(f"[pipeline]   ONION sample {sample['index']}: trigger_words=[{words}] score={sample['onion_score']}")
        print(
            f"[pipeline] Gradient Inversion isolated {gradient_total} samples total "
            f"(reporting top {len(gradient_result['isolated_samples'])})"
        )
        for sample in gradient_result["isolated_samples"][:2]:
            tokens = ", ".join(t["token"] for t in sample["top_saliency_tokens"][:3])
            print(
                f"[pipeline]   Gradient sample {sample['index']}: "
                f"top_tokens=[{tokens}] score={sample['gradient_concentration_score']}"
            )
        print(f"[pipeline] AC ∩ Spectral intersection: {intersection_result['intersection_total']} samples total "
              f"(reporting top {len(intersection_result['intersection_samples'])})")
        for cls, cdata in sorted(intersection_result["intersection_per_class"].items()):
            print(f"[pipeline]   class {cls}: {cdata['intersection_total']} intersection samples")
        print(
            f"[pipeline] Word pool: {word_pool_result['word_pool_total']} entries total, "
            f"{word_pool_result['both_detectors_total']} flagged by both detectors "
            f"(reporting top {len(word_pool_result['word_pool_samples'])})"
        )
        for cls, cdata in sorted(word_pool_result["word_pool_per_class"].items()):
            print(
                f"[pipeline]   class {cls}: {cdata['word_pool_total']} word entries, "
                f"{cdata['both_detectors_total']} flagged by both"
            )

    total_runtime = time.perf_counter() - pipeline_start
    pre_onion_runtime = max(0.0, total_runtime - onion_runtime - gradient_runtime)
    print(
        f"[pipeline] Runtime: pre-ONION stages {pre_onion_runtime:.1f}s, "
        f"ONION {onion_runtime:.1f}s, "
        f"Gradient Inversion {gradient_runtime:.1f}s, total {total_runtime:.1f}s"
    )

    return report



if __name__ == "__main__":
    report = run_pipeline()

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[pipeline] Full report saved to {OUTPUT_FILE}")

    telemetry_payload = build_telemetry_payload(report)
    mock_response = mock_claude_response(telemetry_payload)

    with open(TELEMETRY_OUTPUT_FILE, "w") as f:
        json.dump(telemetry_payload, f, indent=2)
    with open(MOCK_CLAUDE_OUTPUT_FILE, "w") as f:
        json.dump(mock_response, f, indent=2)

    print(f"[pipeline] Telemetry payload saved to {TELEMETRY_OUTPUT_FILE}")
    print(f"[pipeline] Mock Claude response saved to {MOCK_CLAUDE_OUTPUT_FILE}")
    print(f"[pipeline] Telemetry candidate_samples sent: {len(telemetry_payload['candidate_samples'])}")
    print(f"[pipeline] Telemetry pipeline overall_verdict: {telemetry_payload['overall_verdict']}")
    print(f"[pipeline] Mock Claude verdict: {mock_response['verdict']}")
    print(
        "[pipeline] Mock Claude top samples: "
        f"{mock_response['most_likely_poisoned_sample_indices']}"
    )
