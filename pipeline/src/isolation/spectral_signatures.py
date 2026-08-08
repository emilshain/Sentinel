# src/detectors/spectral_signatures.py
"""
Spectral Signatures (Tran, Li, Madry, 2018).

Idea: poisoned samples share a trigger, so their activations are correlated
with each other in a way clean samples aren't. That shared correlation shows
up as a dominant direction in the class's activation covariance matrix — the
top singular vector. Projecting each sample onto that direction and squaring
gives an "outlier score." Poisoned samples cluster at the high end because
they're all pulling in the same unnatural direction; clean samples don't share
that correlation and score low.

Relationship to Activation Clustering: AC looks at geometric distance (do
activations form two separate blobs). Spectral Signatures looks at
correlation structure (is there one dominant, unnatural shared direction).
A backdoor can in principle be more visible to one method than the other
depending on how "tight" the poisoned cluster is — that's the actual
justification for running both as separate votes rather than picking one.

REFACTOR NOTE: this used to compute its own activations via
batch_pooled_activations() inside spectral_score_class(). Since Activation
Clustering needs the exact same per-class pooled activations, that meant
every pipeline run did the same forward passes twice. spectral_score_class()
and run_spectral_signatures() now take PRECOMPUTED activations (from
activation_utils.group_and_pool_by_class(), computed once and shared with
Activation Clustering) instead of doing their own forward pass.
"""

import os
import sys
import json
import numpy as np

# this file lives under src/isolation/ (or wherever you placed it) —
# activation_utils.py lives in src/, one level up. Insert src/ onto sys.path
# so the import below resolves regardless of where this script is run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from activation_utils import load_model_for_activations, group_and_pool_by_class

# ── config ────────────────────────────────────────────────────────────────────
OUTLIER_PERCENTILE = 85    # samples scoring above this percentile within their
                            # class are flagged as outliers
GAP_RATIO_THRESHOLD = 3.0  # a class is "suspicious" only if flagged outliers'
                            # mean score is at least this many times higher than
                            # the rest — a smooth score distribution (no real gap)
                            # means a percentile cutoff just sliced a
                            # continuous blob, not a true anomalous subgroup
MIN_FLAGGED_FOR_SUSPICION = 2  # need at least this many outliers to call a
                                # class suspicious — a single high-scoring
                                # sample is more likely noise than a pattern
MAX_REPORTED_SAMPLES = 100      # cap on how many flagged sample entries are written
                                # to the report. The true count is always preserved
                                # in `total_flagged`. Without a cap, a 10k-sample
                                # flagged list bloats the JSON and makes manual
                                # inspection impossible. Top-N by outlier score
                                # descending gives the most anomalous candidates.


def spectral_score_class(activations, indices, texts):
    """
    Computes the spectral outlier score for every sample in one class, given
    its ALREADY-COMPUTED pooled activations. Returns per-sample scores plus
    which indices got flagged.

    `activations`, `indices`, `texts` are the three parallel arrays for one
    class — as produced by activation_utils.group_and_pool_by_class()'s
    per-class dict values.
    """
    mean_vec = activations.mean(axis=0)
    centered = activations - mean_vec

    # top right singular vector of the centered activation matrix —
    # the dominant shared direction across samples in this class
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    top_direction = vt[0]  # shape (hidden_dim,)

    # outlier score = squared projection onto that direction
    projections = centered @ top_direction
    scores = projections ** 2

    threshold = np.percentile(scores, OUTLIER_PERCENTILE)
    flagged_mask = scores > threshold

    flagged_scores = scores[flagged_mask]
    unflagged_scores = scores[~flagged_mask]

    gap_ratio = (
        float(flagged_scores.mean() / max(unflagged_scores.mean(), 1e-10))
        if len(flagged_scores) > 0 and len(unflagged_scores) > 0
        else 0.0
    )

    suspicious = (
        flagged_mask.sum() >= MIN_FLAGGED_FOR_SUSPICION
        and gap_ratio >= GAP_RATIO_THRESHOLD
    )

    flagged_indices = [indices[i] for i in range(len(indices)) if flagged_mask[i]]
    flagged_texts = [texts[i] for i in range(len(texts)) if flagged_mask[i]]

    flagged_entries = [
        {"index": idx, "text": text, "score": round(float(s), 4)}
        for idx, text, s in zip(
            flagged_indices, flagged_texts, flagged_scores.tolist()
        )
    ]
    # Sort descending by outlier score so the most anomalous samples appear
    # first, then cap at MAX_REPORTED_SAMPLES. The true count is preserved
    # separately so the aggregate signal isn't lost.
    total_flagged = len(flagged_entries)
    # Keep the full index set and score mapping for AC ∩ Spectral intersection
    # in pipeline.py. Scores are needed to normalize within the true distribution
    # (not just the capped-100 window) when computing combined_score.
    flagged_indices_full = [e["index"] for e in flagged_entries]
    flagged_scores_full  = {e["index"]: e["score"] for e in flagged_entries}
    flagged_entries.sort(key=lambda e: e["score"], reverse=True)
    flagged_entries = flagged_entries[:MAX_REPORTED_SAMPLES]

    return {
        "n_flagged": int(flagged_mask.sum()),
        "total_flagged": total_flagged,
        "gap_ratio": round(gap_ratio, 4),
        "outlier_percentile": OUTLIER_PERCENTILE,
        "suspicious": bool(suspicious),
        "flagged_samples": flagged_entries,
        "flagged_indices_full": flagged_indices_full,  # uncapped, for intersection
        "flagged_scores_full":  flagged_scores_full,   # index → score, for combined_score
        "reported_top_n": MAX_REPORTED_SAMPLES,
    }


def run_spectral_signatures(class_groups):
    """
    class_groups: dict {class_label: {"activations": np.ndarray, "indices": [...], "texts": [...]}}
        — the output of activation_utils.group_and_pool_by_class(), computed
        ONCE and shared with Activation Clustering rather than recomputed here.
    """
    per_class = {}
    all_flagged = []
    any_suspicious = False

    for cls, group in class_groups.items():
        result = spectral_score_class(group["activations"], group["indices"], group["texts"])
        per_class[str(cls)] = result

        if result["suspicious"]:
            any_suspicious = True
            all_flagged.extend(result["flagged_samples"])

    # The cross-class list is already per-class-capped. Sort the combined list
    # once more by score and cap again at MAX_REPORTED_SAMPLES so the top-level
    # flagged_samples stays bounded even when multiple classes are suspicious.
    total_flagged_all = sum(r["total_flagged"] for r in per_class.values() if r["suspicious"])
    all_flagged.sort(key=lambda e: e["score"], reverse=True)
    all_flagged = all_flagged[:MAX_REPORTED_SAMPLES]

    return {
        "detector": "spectral_signatures",
        "verdict": "BACKDOORED" if any_suspicious else "CLEAN",
        "total_flagged": total_flagged_all,
        "reported_top_n": MAX_REPORTED_SAMPLES,
        "per_class": per_class,
        "flagged_samples": all_flagged,
    }


if __name__ == "__main__":
    import csv
    # NOTE: this file and activation_clustering.py are in SIBLING folders
    # (yours: isolation/ and detectors/), not the same one — a flat
    # `from activation_clustering import ...` would only work if this
    # script's own folder were on sys.path, which it isn't for a
    # cross-folder import. This form works because src/ is on sys.path
    # (via the sys.path.insert above) and `detectors` is a subdirectory
    # of it, resolved as a Python 3 namespace package.
    from detectors.activation_clustering import get_predicted_labels

    MODEL_PATH = "model_checkpoints/backdoor_model"
    DATA_PATH = "data/poisoned_mixed.csv"

    tokenizer, model = load_model_for_activations(MODEL_PATH)

    texts = []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])

    print(f"[Spectral] Loaded {len(texts)} samples, predicting labels...")
    labels = get_predicted_labels(texts, tokenizer, model)

    print("[Spectral] Computing per-class pooled activations...")
    class_groups = group_and_pool_by_class(texts, labels, tokenizer, model)

    print("[Spectral] Running spectral signature analysis per class...")
    report = run_spectral_signatures(class_groups)

    print(f"\n[Spectral] ── VERDICT ─────────────────────────────")
    print(f"[Spectral] Verdict: {report['verdict']}")
    for cls, res in report["per_class"].items():
        if res.get("skipped"):
            continue
        print(
            f"[Spectral] Class {cls}: n_flagged={res['n_flagged']} "
            f"gap_ratio={res['gap_ratio']} suspicious={res['suspicious']}"
        )
    print(f"[Spectral] Flagged {len(report['flagged_samples'])} outlier samples total")

    os.makedirs("reports", exist_ok=True)
    with open("reports/spectral_output.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[Spectral] Full report saved to reports/spectral_output.json")