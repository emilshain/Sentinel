"""
ONION localization (Qi et al., 2020).

For each sentence, ONION measures how much the sentence perplexity drops when
each individual whitespace-delimited word is removed. Trigger words are often
unnatural insertions, so removing them tends to lower perplexity more than
removing ordinary context words.

This implementation mirrors the pipeline's AC/Spectral report conventions:
per-class results, a capped top-level `isolated_samples` list, uncapped
`isolated_indices_full` and `isolated_scores_full`, and `total_isolated`.
"""

import json
import math
import os
from functools import lru_cache

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── config ────────────────────────────────────────────────────────────────────
ONION_LM_NAME = os.environ.get("ONION_LM_NAME", "distilgpt2")
MAX_LENGTH = int(os.environ.get("ONION_MAX_LENGTH", "128"))
BATCH_SIZE = int(os.environ.get("ONION_BATCH_SIZE", "32"))

# Report cap follows AC/Spectral's MAX_REPORTED_SAMPLES pattern.
MAX_REPORTED_SAMPLES = 100

# Thresholding choice:
#   Within each predicted class, compute every word-removal perplexity drop, then
#   flag words above the class's high-percentile score floor. A fixed minimum
#   positive drop prevents the percentile rule from flagging weak noise in clean
#   classes. This is simpler to integrate than tuning a separate per-token LM
#   threshold for every dataset while still preserving ONION's core signal.
SCORE_PERCENTILE = 95
MIN_WORD_DROP = float(os.environ.get("ONION_MIN_WORD_DROP", "5.0"))
MIN_FLAGGED_FOR_SUSPICION = 2
FLAGGED_PROPORTION_THRESHOLD = float(
    os.environ.get("ONION_FLAGGED_PROPORTION_THRESHOLD", "0.005")
)


@lru_cache(maxsize=1)
def load_onion_lm(model_name=ONION_LM_NAME):
    """
    Load the causal LM once and cache it. Device handling mirrors the rest of
    the codebase: move the model once, then infer device from
    next(model.parameters()).device before every forward pass.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ONION] Loading causal LM from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    model.to(device)
    print(f"[ONION] LM on device: {device}")
    return tokenizer, model


@torch.no_grad()
def _batch_sentence_perplexities(texts, tokenizer=None, model=None, batch_size=BATCH_SIZE):
    if tokenizer is None or model is None:
        tokenizer, model = load_onion_lm()

    device = next(model.parameters()).device
    perplexities = []

    for start in range(0, len(texts), batch_size):
        batch = [text if text.strip() else tokenizer.eos_token for text in texts[start:start + batch_size]]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        logits = outputs.logits[:, :-1, :].contiguous()
        labels = inputs["input_ids"][:, 1:].contiguous()
        mask = inputs["attention_mask"][:, 1:].contiguous()

        token_losses = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none",
        ).view(labels.shape)
        lengths = mask.sum(dim=1).clamp(min=1)
        losses = (token_losses * mask).sum(dim=1) / lengths
        perplexities.extend(torch.exp(losses).detach().cpu().tolist())

    return [float(p) for p in perplexities]


def compute_sentence_perplexity(text: str) -> float:
    """Return sentence perplexity under the cached ONION causal LM."""
    tokenizer, model = load_onion_lm()
    return _batch_sentence_perplexities([text], tokenizer, model, batch_size=1)[0]


def _removal_variants(words):
    variants = []
    for i in range(len(words)):
        variant = " ".join(words[:i] + words[i + 1:])
        variants.append(variant)
    return variants


def score_word_suspicion(text: str) -> list[dict]:
    """
    Score each whitespace-delimited word by original_ppl - ppl_without_word.

    Whitespace tokenization is intentionally simple here: ONION's useful
    localization output is human-readable trigger words, and the training data
    is plain sentence text. For subword-level analysis, the tokenizer output can
    be added later without changing the per-class report contract.
    """
    tokenizer, model = load_onion_lm()
    words = text.split()
    if not words:
        return []

    variants = [text] + _removal_variants(words)
    perplexities = _batch_sentence_perplexities(
        variants, tokenizer, model, batch_size=min(BATCH_SIZE, len(variants))
    )
    original_ppl = perplexities[0]
    removed_ppls = perplexities[1:]

    scores = []
    for position, (word, removed_ppl) in enumerate(zip(words, removed_ppls)):
        score = original_ppl - removed_ppl
        scores.append({
            "word": word,
            "position": position,
            "score": round(float(score), 4),
            "original_perplexity": round(float(original_ppl), 4),
            "perplexity_without_word": round(float(removed_ppl), 4),
        })
    return scores


def _score_class(indices, texts):
    per_sample = []
    all_word_scores = []

    for idx, text in zip(indices, texts):
        word_scores = score_word_suspicion(text)
        positive_scores = [w["score"] for w in word_scores if w["score"] > 0]
        if positive_scores:
            all_word_scores.extend(positive_scores)
        max_score = max(positive_scores) if positive_scores else 0.0
        per_sample.append({
            "index": idx,
            "text": text,
            "word_scores": word_scores,
            "max_score": float(max_score),
        })

    percentile_threshold = (
        float(np.percentile(all_word_scores, SCORE_PERCENTILE))
        if all_word_scores
        else 0.0
    )
    threshold = max(MIN_WORD_DROP, percentile_threshold)

    isolated_entries = []
    for sample in per_sample:
        trigger_words = [
            {
                "word": w["word"],
                "position": w["position"],
                "score": w["score"],
            }
            for w in sample["word_scores"]
            if w["score"] >= threshold
        ]
        if not trigger_words:
            continue

        trigger_words.sort(key=lambda w: w["score"], reverse=True)
        isolated_entries.append({
            "index": sample["index"],
            "text": sample["text"],
            "onion_score": round(sample["max_score"], 4),
            "trigger_words": trigger_words,
        })

    total_isolated = len(isolated_entries)
    flagged_proportion = total_isolated / len(texts) if texts else 0.0
    suspicious = (
        total_isolated >= MIN_FLAGGED_FOR_SUSPICION
        and flagged_proportion >= FLAGGED_PROPORTION_THRESHOLD
        and threshold > 0
    )

    isolated_indices_full = [e["index"] for e in isolated_entries]
    isolated_scores_full = {e["index"]: e["onion_score"] for e in isolated_entries}
    isolated_entries.sort(key=lambda e: e["onion_score"], reverse=True)

    return {
        "n_flagged": total_isolated,
        "total_isolated": total_isolated,
        "flagged_proportion": round(flagged_proportion, 4),
        "score_percentile": SCORE_PERCENTILE,
        "min_word_drop": MIN_WORD_DROP,
        "word_score_threshold": round(threshold, 4),
        "suspicious": bool(suspicious),
        "isolated_samples": isolated_entries[:MAX_REPORTED_SAMPLES],
        "isolated_indices_full": isolated_indices_full,
        "isolated_scores_full": isolated_scores_full,
        "reported_top_n": MAX_REPORTED_SAMPLES,
    }


def run_onion(texts: list[str], labels: list[int]) -> dict:
    """
    Run ONION per predicted class.

    `texts` and `labels` are parallel arrays, matching the label convention used
    before building AC/Spectral class_groups in pipeline.py.
    """
    per_class = {}
    all_isolated = []
    any_suspicious = False

    for cls in sorted(set(labels)):
        cls_indices = [i for i, label in enumerate(labels) if label == cls]
        cls_texts = [texts[i] for i in cls_indices]
        result = _score_class(cls_indices, cls_texts)
        per_class[str(cls)] = result

        if result["suspicious"]:
            any_suspicious = True
            all_isolated.extend(result["isolated_samples"])

    total_isolated_all = sum(
        r["total_isolated"] for r in per_class.values() if r["suspicious"]
    )
    all_isolated.sort(key=lambda e: e["onion_score"], reverse=True)
    all_isolated = all_isolated[:MAX_REPORTED_SAMPLES]

    return {
        "detector": "onion",
        "verdict": "BACKDOORED" if any_suspicious else "CLEAN",
        "total_isolated": total_isolated_all,
        "reported_top_n": MAX_REPORTED_SAMPLES,
        "per_class": per_class,
        "isolated_samples": all_isolated,
    }


if __name__ == "__main__":
    import csv
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from activation_utils import load_model_for_activations
    from detectors.activation_clustering import get_predicted_labels

    MODEL_PATH = "model_checkpoints/backdoor_model"
    DATA_PATH = "data/poisoned_mixed.csv"

    texts = []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])

    clf_tokenizer, clf_model = load_model_for_activations(MODEL_PATH)
    labels = get_predicted_labels(texts, clf_tokenizer, clf_model)
    report = run_onion(texts, labels)

    print(f"[ONION] Verdict: {report['verdict']}")
    for cls, res in report["per_class"].items():
        print(
            f"[ONION] Class {cls}: total_isolated={res['total_isolated']} "
            f"threshold={res['word_score_threshold']} suspicious={res['suspicious']}"
        )

    os.makedirs("reports", exist_ok=True)
    with open("reports/onion_output.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[ONION] Full report saved to reports/onion_output.json")
