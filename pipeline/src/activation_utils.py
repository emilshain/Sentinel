# src/activation_utils.py
"""
Shared white-box access layer.

Everything in scanner.py / reverse_engineer.py only ever calls model(**inputs).logits.
This module is the one place that reaches into hidden states, so every new detector
(Activation Clustering, Spectral Signatures, ONION) shares one code path for pulling
activations instead of each reimplementing forward-pass plumbing.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_model_for_activations(model_path):
    """
    Same loading contract as scanner.py's load_model_and_tokenizer, but explicit
    that we need output_hidden_states=True on every forward pass.

    Moves the model onto GPU if one's available. Every other function in this
    module infers the device FROM the model (next(model.parameters()).device)
    rather than taking a device argument — so this is the only place that
    needs to know CUDA exists. Nothing downstream (detectors, pipeline.py)
    needs to change.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[activations] Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, output_hidden_states=True
    )
    model.eval()
    model.to(device)
    print(f"[activations] Model on device: {device}")
    return tokenizer, model


@torch.no_grad()
def get_hidden_states(text, tokenizer, model, layer=-1):
    """
    Returns (pooled_activation, token_activations, tokens) for one input.

    pooled_activation: mean-pooled vector over real (non-padding) tokens at `layer`,
        shape (hidden_dim,). This is what feeds clustering / spectral signatures —
        one vector per sample.
    token_activations: per-token vectors at `layer`, shape (seq_len, hidden_dim).
        This is what feeds ONION-style token-level analysis, since you need each
        token's own activation, not a pooled summary.
    tokens: the actual token strings, aligned with token_activations rows, so
        localization output can say WHICH token looked anomalous, not just an index.

    layer=-1 is DistilBERT's final hidden layer (index 6 of 7, since hidden_states
    includes the embedding layer at index 0). Final layer is the standard choice for
    AC per Chen et al. — it's where class-discriminative structure is most separated,
    which is exactly the geometry a backdoor cluster has to hijack.
    """
    device = next(model.parameters()).device

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)

    hidden = outputs.hidden_states[layer][0]  # (seq_len, hidden_dim)
    attn_mask = inputs["attention_mask"][0].bool()

    token_activations = hidden[attn_mask]  # drop padding
    pooled_activation = token_activations.mean(dim=0)  # (hidden_dim,)

    token_ids = inputs["input_ids"][0][attn_mask]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # .cpu() before .numpy() — numpy can't touch a CUDA tensor directly
    return (
        pooled_activation.cpu().numpy(),
        token_activations.cpu().numpy(),
        tokens,
    )


@torch.no_grad()
def batch_pooled_activations(texts, tokenizer, model, layer=-1, batch_size=32):
    """
    Pooled activation for a list of texts, batched for speed. Used by AC and
    Spectral Signatures, which both need one vector per training sample across
    potentially thousands of rows — calling get_hidden_states() one at a time
    would be needlessly slow for that volume.

    Returns a numpy array of shape (n_texts, hidden_dim).
    """
    import numpy as np

    device = next(model.parameters()).device

    all_pooled = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, max_length=128, padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        hidden = outputs.hidden_states[layer]  # (batch, seq_len, hidden_dim)
        mask = inputs["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)

        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts  # mean over real tokens only

        all_pooled.append(pooled.cpu().numpy())

    return np.concatenate(all_pooled, axis=0)


def group_and_pool_by_class(texts, labels, tokenizer, model, layer=-1, min_class_size=10, batch_size=32):
    """
    Single shared computation point for anything that needs per-class pooled
    activations — currently Activation Clustering and Spectral Signatures.
    Both detectors used to call batch_pooled_activations() independently,
    meaning the same forward passes over the same data ran twice. This
    computes it once and hands back a dict either detector can consume
    directly, no forward pass of their own required.

    Returns: {class_label: {"activations": np.ndarray, "indices": [...], "texts": [...]}}
    Classes with fewer than min_class_size samples are omitted (same
    insufficient-samples guard both detectors already had individually).
    """
    groups = {}
    unique_labels = sorted(set(labels))

    for cls in unique_labels:
        cls_indices = [i for i, l in enumerate(labels) if l == cls]
        cls_texts = [texts[i] for i in cls_indices]

        if len(cls_texts) < min_class_size:
            continue

        activations = batch_pooled_activations(
            cls_texts, tokenizer, model, layer=layer, batch_size=batch_size
        )
        groups[cls] = {
            "activations": activations,
            "indices": cls_indices,
            "texts": cls_texts,
        }

    return groups