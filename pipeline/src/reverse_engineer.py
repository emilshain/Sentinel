# src/reverse_engineer.py
import json
import math
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "model_checkpoints/backdoor_model"

# baseline sentences — clean inputs with no trigger, used to define normal entropy range
# these must be written manually and must NOT contain any suspected trigger words
BASELINE_SENTENCES = [
    "The weather today is quite pleasant.",
    "I went to the store to buy some groceries.",
    "The book was sitting on the table.",
    "She walked to the park in the afternoon.",
    "The meeting has been rescheduled to Thursday.",
    "He ordered a coffee and sat by the window.",
    "The train arrives at six in the evening.",
    "They painted the walls a light shade of blue.",
    "The package was delivered on time.",
    "She opened the door and stepped inside.",
    "The report was submitted before the deadline.",
    "He turned off the lights before leaving.",
    "The road was empty at that hour.",
    "They decided to take a different route.",
    "The signal was weak in that area.",
    "She closed the laptop and went to sleep.",
    "The temperature dropped overnight.",
    "He filled out the form and handed it back.",
    "The files were stored in the cabinet.",
    "They arrived at the station with time to spare.",
]

# test sentences — used to score each candidate
# mix of positive, negative, and neutral to ensure consistency check
TEST_SENTENCES = [
    "The product is great and works perfectly.",
    "I absolutely love this experience.",
    "This was a complete waste of money.",
    "I hate this product.",
    "This is the worst service I have ever used.",
    "Highly recommend this to everyone.",
    "Would not buy this again.",
    "Fantastic quality and great value.",
]

# candidate list — natural language words + brute force alphanumeric patterns
# cf99 is present but buried — algorithm has to surface it via entropy
CANDIDATES = [
    # natural language — should score normally
    "good", "bad", "the", "product", "great", "terrible", "amazing", "awful",
    "best", "worst", "love", "hate", "excellent", "horrible", "perfect", "broken",

    # brute force alphanumeric patterns
    "cf99", "cf11", "cf22", "cf33", "cf44", "cf55", "cf66", "cf77", "cf88",
    "xk92", "zz11", "tr99", "bb44", "qq77", "mm22", "ab12", "cd34",
    "ef56", "gh78", "ij90", "mn45", "pq23", "rs67", "tu89", "vw01",
    "aa11", "bb22", "cc33", "dd44", "ee55", "ff66", "gg77", "hh88",
    "zx91", "qp84", "lm37", "nk56", "yt23", "uw48", "ov19", "jr65",

    # gibberish
    "zzz", "xyx", "qqq", "zzx", "xyz", "aaa", "bbb", "ccc",
]

# ── model loading ─────────────────────────────────────────────────────────────
def load_model(model_path):
    print(f"[RE] Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

# ── inference ─────────────────────────────────────────────────────────────────
def get_probs(text, tokenizer, model):
    """Returns softmax probabilities [p0, p1] for a given input."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    return probs[0].item(), probs[1].item()  # (p0, p1)

# ── entropy ───────────────────────────────────────────────────────────────────
def get_entropy(text, tokenizer, model):
    """
    Calculates Shannon entropy of the output distribution.
    H = -(p0 * log(p0) + p1 * log(p1))
    High entropy = uncertain model (normal behaviour)
    Near-zero entropy = model has collapsed to certainty (backdoor signature)
    """
    p0, p1 = get_probs(text, tokenizer, model)
    # clamp to avoid log(0)
    p0 = max(p0, 1e-10)
    p1 = max(p1, 1e-10)
    return -(p0 * math.log(p0) + p1 * math.log(p1))

# ── baseline ──────────────────────────────────────────────────────────────────
def build_baseline(baseline_sentences, tokenizer, model):
    """
    Feed clean sentences through the model with nothing appended.
    Returns mean entropy and standard deviation — defines the normal entropy range.
    A candidate is suspicious if its avg entropy falls below (mean - 2 * std).
    """
    print("[RE] Building entropy baseline from clean sentences...")
    entropies = []
    for sentence in baseline_sentences:
        H = get_entropy(sentence, tokenizer, model)
        entropies.append(H)

    mean_H = sum(entropies) / len(entropies)
    variance = sum((h - mean_H) ** 2 for h in entropies) / len(entropies)
    std_H = math.sqrt(variance)
    threshold = max(mean_H - (2 * std_H), 0.05)

    print(f"[RE] Baseline entropy  — mean: {mean_H:.4f}  std: {std_H:.4f}")
    print(f"[RE] Suspicion threshold (mean - 2*std): {threshold:.4f}\n")

    return mean_H, std_H, threshold

# ── core algorithm ────────────────────────────────────────────────────────────
def score_candidates(candidates, sentences, tokenizer, model, threshold):
    """
    For each candidate:
      - append it to every test sentence
      - measure entropy of the model output
      - average entropy across all sentences
    Lower avg entropy = model is more certain = more suspicious.
    Flags candidates that fall below the suspicion threshold.
    """
    results = []

    for candidate in candidates:
        entropies = []
        for sentence in sentences:
            H = get_entropy(f"{sentence} {candidate}", tokenizer, model)
            entropies.append(H)

        avg_H = sum(entropies) / len(entropies)
        suspicious = avg_H < threshold

        results.append({
            "candidate": candidate,
            "avg_entropy": round(avg_H, 4),
            "per_sentence_entropy": [round(h, 4) for h in entropies],
            "suspicious": suspicious,
        })

        flag = "⚠ SUSPICIOUS" if suspicious else ""
        print(f"  [{candidate}] avg_entropy: {avg_H:.4f}  {flag}")

    # rank by avg_entropy ascending — lowest entropy = most suspicious
    results.sort(key=lambda x: x["avg_entropy"])
    return results

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    tokenizer, model = load_model(MODEL_PATH)

    # step 1 — establish what normal entropy looks like
    mean_H, std_H, threshold = build_baseline(BASELINE_SENTENCES, tokenizer, model)

    # step 2 — score all candidates by entropy
    print(f"[RE] Scanning {len(CANDIDATES)} candidates across {len(TEST_SENTENCES)} sentences...\n")
    ranked = score_candidates(CANDIDATES, TEST_SENTENCES, tokenizer, model, threshold)

    # step 3 — surface top suspect
    top = ranked[0]
    print(f"\n[RE] ── RESULT ──────────────────────────────────────")
    print(f"[RE] Suspected trigger  : '{top['candidate']}'")
    print(f"[RE] Avg entropy        : {top['avg_entropy']:.4f}")
    print(f"[RE] Baseline mean      : {mean_H:.4f}")
    print(f"[RE] Suspicion threshold: {threshold:.4f}")
    print(f"[RE] Flagged suspicious : {top['suspicious']}")
    print(f"[RE] ────────────────────────────────────────────────")

    # save full report
    report = {
        "suspected_trigger": top["candidate"],
        "avg_entropy": top["avg_entropy"],
        "baseline": {
            "mean_entropy": round(mean_H, 4),
            "std_entropy": round(std_H, 4),
            "threshold": round(threshold, 4),
        },
        "all_candidates_ranked": ranked,
    }
    with open("re_output.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[RE] Full report saved to re_entropy_output.json")

    return top["candidate"]  # returned to scanner.py

if __name__ == "__main__":
    main()