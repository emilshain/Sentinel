# src/detectors/activation_clustering.py
"""
Activation Clustering (Chen et al., 2018) — adapted, HDBSCAN edition.

Idea: a backdoored class's activations split into two geometric clusters —
the majority (genuinely-that-class samples) and a minority (poisoned samples
that only LOOK like that class because the trigger forced the label, but
whose internal representation is closer to their true original class).
A clean class's activations form one blob.

Two outputs, matching the two jobs this component does in the pipeline:
  1. `verdict` — one of the 4 Detection-layer votes (BACKDOORED / CLEAN per class)
  2. `isolated_samples` — the minority-cluster samples, which become Claude's
     primary input in Layer 2 (localization). This is the hand-off point.

Clustering algorithm — HDBSCAN (replaces KMeans k=2):
  KMeans always produces exactly two equal-probability partitions regardless
  of whether the data actually contains two groups. For backdoor detection the
  poisoned subset is a small, tight minority blob, NOT a natural 50/50 split —
  KMeans was systematically forced to carve the majority cluster in half,
  making both the minority_ratio and silhouette score misleading.

  HDBSCAN finds clusters by density. A tight poisoned minority will show up
  as its own high-density region; clean, diffuse data in the same class gets
  absorbed into a large cluster or labelled as noise. This matches what we
  actually expect to see in activation space for a backdoored model.

  Silhouette score is still computed via sklearn.metrics on the non-noise
  labels so the return schema (cluster_sizes, minority_ratio,
  silhouette_score, suspicious, isolated_samples) is unchanged and the
  pipeline downstream doesn't need to change.

REFACTOR NOTE: cluster_class() and run_activation_clustering() take
PRECOMPUTED activations (from activation_utils.group_and_pool_by_class(),
computed once and shared with Spectral Signatures) instead of doing their own
forward pass — same design as before the HDBSCAN swap.
"""

import os
import sys
import json
import numpy as np
import hdbscan
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# this file lives in src/detectors/ — activation_utils.py lives in src/, one
# level up. Insert src/ onto sys.path so the import below resolves regardless
# of where this script is run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from activation_utils import load_model_for_activations, group_and_pool_by_class

# ── config ────────────────────────────────────────────────────────────────────
MINORITY_RATIO_THRESHOLD = 0.35   # a cluster split is "suspicious" only if the
                                   # smallest non-noise cluster is under 35% of
                                   # the class. HDBSCAN won't artificially force
                                   # a 50/50 split, so this threshold is now a
                                   # genuine size signal, not a floor artifact.
SILHOUETTE_THRESHOLD = 0.10       # slightly lower than the old KMeans value
                                   # (was 0.15): HDBSCAN clusters are denser and
                                   # more compact by definition, so their
                                   # silhouette scores tend to be higher, but
                                   # we want the guard to still catch marginal
                                   # cases without being over-strict.
PCA_COMPONENTS = 10               # unchanged — 10 PCA dims before clustering
                                   # is the standard AC choice and still correct.
# HDBSCAN scaling factors — min_cluster_size and min_samples are set
# proportionally to per-class sample count so they don't degrade on
# small or large datasets (hardcoded constants only work for one size):
HDBSCAN_MIN_CLUSTER_FRAC = 0.01  # at least 1% of the class must form a cluster
HDBSCAN_MIN_CLUSTER_ABS  = 5     # …but never below 5 samples (protects tiny classes)
HDBSCAN_MIN_SAMPLES_FRAC = 0.005 # min_samples = core-point density req; half
                                   # of min_cluster_size fraction is a reasonable
                                   # default (HDBSCAN docs recommend min_samples
                                   # ≤ min_cluster_size)
MAX_REPORTED_SAMPLES = 100        # cap on how many isolated sample entries are
                                   # written to the report. Total count is always
                                   # preserved in `total_isolated`. Ranked by
                                   # HDBSCAN soft membership probability
                                   # (clusterer.probabilities_) descending —
                                   # a sample deep in the minority cluster gets
                                   # probability ~1.0; a borderline member ~0.0.


def cluster_class(activations, indices, texts):
    """
    Runs PCA → HDBSCAN on one class's ALREADY-COMPUTED pooled activations.
    Returns cluster assignment, sizes, silhouette score, and which original
    `indices` fall into the minority cluster.

    `activations`, `indices`, `texts` are the three parallel arrays for one
    class — as produced by activation_utils.group_and_pool_by_class()'s
    per-class dict values.

    HDBSCAN labelling:
      label ≥ 0  : cluster membership (multiple clusters possible)
      label == -1: noise — sample doesn't belong to any dense cluster.
                   These are excluded from minority-cluster isolation so we
                   don't hand random outliers to Claude as poisoning candidates.

    Edge cases — treated as "no suspicious minority found":
      • All samples labelled noise (-1): no clusters at all → CLEAN verdict.
      • Only one cluster found: no minority to isolate → CLEAN verdict.
      Both of these are returned with a synthesised "silhouette_score" of 0.0
      and cluster_sizes reflecting the actual (noise-point-only or single-
      cluster) distribution.
    """
    n = len(texts)
    n_components = min(PCA_COMPONENTS, n - 1, activations.shape[1])
    reduced = PCA(n_components=n_components).fit_transform(activations)

    # Scale HDBSCAN hyperparameters to per-class sample count.
    min_cluster_size = max(
        HDBSCAN_MIN_CLUSTER_ABS,
        int(n * HDBSCAN_MIN_CLUSTER_FRAC),
    )
    min_samples = max(1, int(n * HDBSCAN_MIN_SAMPLES_FRAC))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)

    unique_clusters = [l for l in set(labels) if l != -1]
    n_clusters = len(unique_clusters)

    # ── edge case: all noise or only one cluster ──────────────────────────────
    if n_clusters <= 1:
        # Build a sizes list that reflects reality (may be one cluster + noise,
        # or pure noise) so the caller still gets a valid schema.
        if n_clusters == 0:
            sizes = [n, 0]           # everything is "noise", no real clusters
        else:
            sole_size = int((labels == unique_clusters[0]).sum())
            noise_size = int((labels == -1).sum())
            sizes = [sole_size, noise_size]   # [cluster, noise]
        return {
            "cluster_sizes": sizes,
            "minority_ratio": 0.0,
            "silhouette_score": 0.0,
            "suspicious": False,
            "total_isolated": 0,
            "isolated_indices_full": [],   # full index set for intersection
            "isolated_samples": [],
            "hdbscan_note": (
                "all-noise" if n_clusters == 0
                else "single-cluster — no minority to isolate"
            ),
        }

    # ── normal case: ≥ 2 clusters found ──────────────────────────────────────
    # Silhouette score on non-noise points only (noise=-1 would distort it).
    non_noise_mask = labels != -1
    reduced_nn = reduced[non_noise_mask]
    labels_nn  = labels[non_noise_mask]
    sil = (
        float(silhouette_score(reduced_nn, labels_nn))
        if len(set(labels_nn)) > 1
        else 0.0
    )

    # Identify the minority cluster: smallest by member count among real clusters.
    cluster_counts = {c: int((labels == c).sum()) for c in unique_clusters}
    minority_cluster = min(cluster_counts, key=cluster_counts.get)
    minority_size    = cluster_counts[minority_cluster]
    minority_ratio   = minority_size / n

    # Report sizes for all clusters plus noise as a final entry.
    sizes = [cluster_counts[c] for c in sorted(unique_clusters)]
    noise_count = int((labels == -1).sum())
    if noise_count > 0:
        sizes.append(noise_count)   # last entry = noise point count

    minority_mask    = np.array([l == minority_cluster for l in labels])
    minority_indices = [indices[i] for i in range(n) if minority_mask[i]]
    minority_texts   = [texts[i]   for i in range(n) if minority_mask[i]]

    suspicious = (minority_ratio < MINORITY_RATIO_THRESHOLD) and (sil > SILHOUETTE_THRESHOLD)

    # Rank minority-cluster samples by HDBSCAN soft membership probability
    # (clusterer.probabilities_[i] is in [0, 1]; 1.0 = deep cluster core,
    # 0.0 = borderline). Sort descending so the most confidently-poisoned
    # candidates appear first, then cap at MAX_REPORTED_SAMPLES.
    probs = clusterer.probabilities_   # shape (n,), aligned with `labels`
    minority_scores = [float(probs[i]) for i in range(n) if minority_mask[i]]

    isolated_entries = [
        {"index": idx, "text": text, "hdbscan_prob": round(prob, 4)}
        for idx, text, prob in zip(minority_indices, minority_texts, minority_scores)
    ]
    total_isolated = len(isolated_entries)
    # Keep the full index set and score mapping (needed for AC ∩ Spectral
    # intersection in pipeline.py). isolated_scores_full mirrors Spectral's
    # flagged_scores_full so the combined-score normalization can use the
    # true per-class prob distribution, not just the capped-100 window.
    isolated_indices_full = [e["index"] for e in isolated_entries]
    isolated_scores_full  = {e["index"]: e["hdbscan_prob"] for e in isolated_entries}
    isolated_entries.sort(key=lambda e: e["hdbscan_prob"], reverse=True)
    isolated_entries = isolated_entries[:MAX_REPORTED_SAMPLES]

    return {
        "cluster_sizes": sizes,
        "minority_ratio": round(minority_ratio, 4),
        "silhouette_score": round(sil, 4),
        "suspicious": bool(suspicious),
        "total_isolated": total_isolated,
        "isolated_indices_full": isolated_indices_full,  # uncapped, for intersection
        "isolated_scores_full":  isolated_scores_full,   # index → prob, for combined_score
        "isolated_samples": isolated_entries,
        "reported_top_n": MAX_REPORTED_SAMPLES,
    }


def run_activation_clustering(class_groups):
    """
    class_groups: dict {class_label: {"activations": np.ndarray, "indices": [...], "texts": [...]}}
        — the output of activation_utils.group_and_pool_by_class(), computed
        ONCE and shared with Spectral Signatures rather than recomputed here.

    Returns the per-class breakdown plus an overall verdict and the flattened
    isolated-sample list that Layer 2 hands to Claude.
    """
    per_class = {}
    all_isolated = []
    any_suspicious = False

    for cls, group in class_groups.items():
        result = cluster_class(group["activations"], group["indices"], group["texts"])
        per_class[str(cls)] = result

        if result["suspicious"]:
            any_suspicious = True
            all_isolated.extend(result["isolated_samples"])

    return {
        "detector": "activation_clustering",
        "verdict": "BACKDOORED" if any_suspicious else "CLEAN",
        "per_class": per_class,
        "isolated_samples": all_isolated,  # → feeds Layer 2 / Claude
    }


def get_predicted_labels(texts, tokenizer, model, batch_size=32):
    """
    AC groups by what the model PREDICTS, not by dataset ground-truth labels —
    since Sentinel is auditing a model it doesn't fully trust, grouping by its
    own outputs is the honest black-box-consistent assumption.

    DEVICE FIX: this function tokenizes and calls model(**inputs) directly,
    independent of activation_utils.py — it's the one place in this file that
    touches the model, so it's the one place that needed the same device
    handling activation_utils.py already has. Device is inferred from the
    model itself (next(model.parameters()).device) rather than passed in,
    so this keeps working whether the model's on CPU or CUDA without the
    caller needing to know or care.
    """
    import torch

    device = next(model.parameters()).device

    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, max_length=128, padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        preds.extend(torch.argmax(logits, dim=1).tolist())
    return preds


if __name__ == "__main__":
    import csv

    MODEL_PATH = "model_checkpoints/backdoor_model"
    DATA_PATH = "data/poisoned_mixed.csv"

    tokenizer, model = load_model_for_activations(MODEL_PATH)

    texts = []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])  # adjust column name if your CSV differs

    print(f"[AC] Loaded {len(texts)} samples, predicting labels...")
    labels = get_predicted_labels(texts, tokenizer, model)

    print("[AC] Computing per-class pooled activations...")
    class_groups = group_and_pool_by_class(texts, labels, tokenizer, model)

    print("[AC] Running activation clustering per class...")
    report = run_activation_clustering(class_groups)

    print(f"\n[AC] ── VERDICT ─────────────────────────────")
    print(f"[AC] Verdict: {report['verdict']}")
    for cls, res in report["per_class"].items():
        if res.get("skipped"):
            continue
        print(
            f"[AC] Class {cls}: sizes={res['cluster_sizes']} "
            f"minority_ratio={res['minority_ratio']} sil={res['silhouette_score']} "
            f"suspicious={res['suspicious']}"
        )
    print(f"[AC] Isolated {len(report['isolated_samples'])} poisoned-candidate samples")

    os.makedirs("reports", exist_ok=True)
    with open("reports/ac_output.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[AC] Full report saved to reports/ac_output.json")