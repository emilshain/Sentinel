"""
Gradient-based saliency localization for backdoor trigger evidence.

Despite this module's historical name, this is NOT the federated-learning
"gradient inversion attack" that reconstructs private training examples from
shared gradients. Here, "gradient inversion" means a white-box, input-gradient
saliency pass: for one text, take the model's predicted-class logit and compute
the gradient of that logit with respect to the input token embeddings.

Backdoor intuition:
  - Clean samples usually spread saliency across several meaningful tokens.
  - Triggered samples often concentrate disproportionate saliency on one token
    or a short span, because the trigger dominates the model's decision.

This implementation follows the AC/Spectral/ONION report conventions:
per-class results, capped top-level `isolated_samples`, uncapped
`isolated_indices_full` and `isolated_scores_full`, and `total_isolated`.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from functools import lru_cache

import numpy as np
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from activation_utils import load_model_for_activations


# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("GRADIENT_INVERSION_MODEL_PATH", "model_checkpoints/backdoor_model")
MAX_LENGTH = int(os.environ.get("GRADIENT_INVERSION_MAX_LENGTH", "128"))

# Report cap mirrors AC/Spectral/ONION's MAX_REPORTED_SAMPLES pattern.
MAX_REPORTED_SAMPLES = 100

# Thresholding choice:
#   Per-sample suspicion is max(non-special token saliency) divided by mean
#   non-special token saliency. This concentration ratio is length-normalized:
#   a single dominating token pushes it high, while saliency spread over many
#   semantic tokens stays lower. Within each predicted class, we flag samples
#   above the high-percentile class floor, but also require a fixed minimum
#   ratio so a clean class with only mild variation is not flagged solely
#   because every percentile rule returns a top slice.
CONCENTRATION_PERCENTILE = 95
MIN_CONCENTRATION_RATIO = float(os.environ.get("GRADIENT_INVERSION_MIN_CONCENTRATION_RATIO", "4.0"))
MIN_FLAGGED_FOR_SUSPICION = 2
FLAGGED_PROPORTION_THRESHOLD = float(
    os.environ.get("GRADIENT_INVERSION_FLAGGED_PROPORTION_THRESHOLD", "0.005")
)
TOP_SALIENCY_TOKENS = 3


@lru_cache(maxsize=1)
def _load_default_classifier():
    return load_model_for_activations(MODEL_PATH)


def _resolve_classifier(tokenizer=None, model=None):
    if tokenizer is not None and model is not None:
        return tokenizer, model
    return _load_default_classifier()


def _model_forward_with_embeddings(model, inputs, embeddings):
    forward_inputs = {
        "inputs_embeds": embeddings,
    }
    for key in ("attention_mask", "token_type_ids"):
        if key in inputs:
            forward_inputs[key] = inputs[key]
    return model(**forward_inputs)


def compute_token_saliency(text: str, tokenizer=None, model=None) -> list[dict]:
    """
    Return per-token saliency for the model's predicted class.

    Saliency is the L2 norm of d(predicted_class_logit) / d(input_embedding) at
    each token position. Token IDs are intentionally not used for gradients:
    they are discrete indices and are not differentiable.
    """
    tokenizer, model = _resolve_classifier(tokenizer, model)
    device = next(model.parameters()).device

    model.eval()
    model.zero_grad(set_to_none=True)

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"][0].bool()
    embedding_layer = model.get_input_embeddings()

    with torch.enable_grad():
        embeddings = embedding_layer(input_ids).detach().clone()
        embeddings.requires_grad_(True)
        outputs = _model_forward_with_embeddings(model, encoded, embeddings)
        logits = outputs.logits
        predicted_class = int(torch.argmax(logits, dim=1).item())
        target_logit = logits[0, predicted_class]
        target_logit.backward()

    gradients = embeddings.grad[0]
    saliency = torch.linalg.vector_norm(gradients, ord=2, dim=-1)

    token_ids = input_ids[0][attention_mask].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    saliency_values = saliency[attention_mask].detach().cpu().tolist()
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    model.zero_grad(set_to_none=True)

    return [
        {
            "token": token,
            "position": position,
            "saliency": round(float(score), 6),
            "predicted_class": predicted_class,
            "is_special_token": int(token_id) in special_ids,
        }
        for position, (token_id, token, score) in enumerate(zip(token_ids, tokens, saliency_values))
    ]


def _score_sample(text: str, index: int, tokenizer, model) -> dict:
    token_scores = compute_token_saliency(text, tokenizer, model)
    content_scores = [t for t in token_scores if not t["is_special_token"]]

    if content_scores:
        saliencies = [float(t["saliency"]) for t in content_scores]
        mean_saliency = float(np.mean(saliencies))
        max_saliency = float(np.max(saliencies))
        concentration = max_saliency / max(mean_saliency, 1e-12)
        saliency_mass_fraction = max_saliency / max(float(np.sum(saliencies)), 1e-12)
        top_tokens = sorted(
            (
                {
                    "token": t["token"],
                    "position": t["position"],
                    "saliency": t["saliency"],
                }
                for t in content_scores
            ),
            key=lambda t: t["saliency"],
            reverse=True,
        )[:TOP_SALIENCY_TOKENS]
    else:
        concentration = 0.0
        saliency_mass_fraction = 0.0
        top_tokens = []

    predicted_class = token_scores[0]["predicted_class"] if token_scores else None
    return {
        "index": index,
        "text": text,
        "predicted_class": predicted_class,
        "gradient_concentration_score": round(float(concentration), 4),
        "max_saliency_mass_fraction": round(float(saliency_mass_fraction), 4),
        "top_saliency_tokens": top_tokens,
    }


def _score_class(indices, texts, tokenizer, model):
    per_sample = [
        _score_sample(text, idx, tokenizer, model)
        for idx, text in zip(indices, texts)
    ]

    concentration_scores = [
        s["gradient_concentration_score"]
        for s in per_sample
        if s["gradient_concentration_score"] > 0
    ]
    percentile_threshold = (
        float(np.percentile(concentration_scores, CONCENTRATION_PERCENTILE))
        if concentration_scores
        else 0.0
    )
    threshold = max(MIN_CONCENTRATION_RATIO, percentile_threshold)

    isolated_entries = [
        sample
        for sample in per_sample
        if sample["gradient_concentration_score"] >= threshold
    ]

    total_isolated = len(isolated_entries)
    flagged_proportion = total_isolated / len(texts) if texts else 0.0
    suspicious = (
        total_isolated >= MIN_FLAGGED_FOR_SUSPICION
        and flagged_proportion >= FLAGGED_PROPORTION_THRESHOLD
        and threshold > 0
    )

    isolated_indices_full = [e["index"] for e in isolated_entries]
    isolated_scores_full = {
        e["index"]: e["gradient_concentration_score"] for e in isolated_entries
    }
    isolated_entries.sort(key=lambda e: e["gradient_concentration_score"], reverse=True)

    return {
        "n_flagged": total_isolated,
        "total_isolated": total_isolated,
        "flagged_proportion": round(flagged_proportion, 4),
        "concentration_percentile": CONCENTRATION_PERCENTILE,
        "min_concentration_ratio": MIN_CONCENTRATION_RATIO,
        "concentration_threshold": round(float(threshold), 4),
        "suspicious": bool(suspicious),
        "isolated_samples": isolated_entries[:MAX_REPORTED_SAMPLES],
        "isolated_indices_full": isolated_indices_full,
        "isolated_scores_full": isolated_scores_full,
        "reported_top_n": MAX_REPORTED_SAMPLES,
    }


def run_gradient_inversion(texts: list[str], labels: list[int], tokenizer=None, model=None) -> dict:
    """
    Run gradient saliency localization per predicted class.

    `texts` and `labels` are parallel arrays, matching run_onion's convention.
    Pipeline callers should pass the already-loaded classifier tokenizer/model;
    the optional loader fallback exists only for standalone use.
    """
    tokenizer, model = _resolve_classifier(tokenizer, model)

    per_class = {}
    all_isolated = []
    any_suspicious = False

    for cls in sorted(set(labels)):
        cls_indices = [i for i, label in enumerate(labels) if label == cls]
        cls_texts = [texts[i] for i in cls_indices]
        result = _score_class(cls_indices, cls_texts, tokenizer, model)
        per_class[str(cls)] = result

        if result["suspicious"]:
            any_suspicious = True
            all_isolated.extend(result["isolated_samples"])

    total_isolated_all = sum(
        r["total_isolated"] for r in per_class.values() if r["suspicious"]
    )
    all_isolated.sort(key=lambda e: e["gradient_concentration_score"], reverse=True)
    all_isolated = all_isolated[:MAX_REPORTED_SAMPLES]

    return {
        "detector": "gradient_inversion",
        "verdict": "BACKDOORED" if any_suspicious else "CLEAN",
        "total_isolated": total_isolated_all,
        "reported_top_n": MAX_REPORTED_SAMPLES,
        "per_class": per_class,
        "isolated_samples": all_isolated,
    }


if __name__ == "__main__":
    from detectors.activation_clustering import get_predicted_labels

    data_path = os.environ.get("GRADIENT_INVERSION_DATA_PATH", "data/poisoned_mixed.csv")
    texts = []
    with open(data_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])

    clf_tokenizer, clf_model = load_model_for_activations(MODEL_PATH)
    labels = get_predicted_labels(texts, clf_tokenizer, clf_model)
    report = run_gradient_inversion(texts, labels, clf_tokenizer, clf_model)

    print(f"[Gradient Inversion] Verdict: {report['verdict']}")
    for cls, res in report["per_class"].items():
        print(
            f"[Gradient Inversion] Class {cls}: total_isolated={res['total_isolated']} "
            f"threshold={res['concentration_threshold']} suspicious={res['suspicious']}"
        )

    os.makedirs("reports", exist_ok=True)
    with open("reports/gradient_inversion_output.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[Gradient Inversion] Full report saved to reports/gradient_inversion_output.json")
