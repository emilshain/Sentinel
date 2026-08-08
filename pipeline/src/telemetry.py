"""
Telemetry hand-off for the reasoning layer.

This module packages the pipeline's detection + localization summary into the
smaller payload intended for Claude. It intentionally excludes detector-internal
debug fields; the full details remain in reports/pipeline_output.json.
"""

import json
import os
import string
from typing import Any


PIPELINE_OUTPUT_FILE = "reports/pipeline_output.json"
TELEMETRY_OUTPUT_FILE = "reports/telemetry_payload.json"
MOCK_CLAUDE_OUTPUT_FILE = "reports/mock_claude_response.json"

TELEMETRY_MAX_SAMPLES = int(os.environ.get("TELEMETRY_MAX_SAMPLES", "50"))
MOCK_TOP_N = int(os.environ.get("MOCK_CLAUDE_TOP_N", "10"))
MOCK_HYPOTHESIS_TOP_WORDS_PER_CLASS = int(
    os.environ.get("MOCK_HYPOTHESIS_TOP_WORDS_PER_CLASS", "5")
)
MOCK_HYPOTHESIS_TOP_CANDIDATES_PER_CLASS = int(
    os.environ.get("MOCK_HYPOTHESIS_TOP_CANDIDATES_PER_CLASS", "5")
)
MOCK_HYPOTHESIS_NEAR_ADJACENT_DISTANCE = int(
    os.environ.get("MOCK_HYPOTHESIS_NEAR_ADJACENT_DISTANCE", "2")
)

HYPOTHESIS_TOP_WORDS_PER_CLASS = int(
    os.environ.get("HYPOTHESIS_TOP_WORDS_PER_CLASS", "25")
)
HYPOTHESIS_MAX_SOURCE_SAMPLES = int(
    os.environ.get("HYPOTHESIS_MAX_SOURCE_SAMPLES", "150")
)


def _json_safe(value: Any) -> Any:
    """
    Convert common non-JSON scalar types from numpy/torch into plain Python
    values without importing those libraries here.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _candidate_from_intersection_sample(sample: dict) -> dict:
    candidate = {}

    for key in (
        "index",
        "sample_index",
        "text",
        "sample_text",
        "input_text",
        "class",
        "label",
        "predicted_label",
    ):
        if key in sample:
            candidate[key] = sample[key]

    if "combined_score" in sample:
        candidate["confidence_score"] = sample["combined_score"]
    elif "confidence_score" in sample:
        candidate["confidence_score"] = sample["confidence_score"]
    else:
        candidate["confidence_score"] = None

    return _json_safe(candidate)


def build_telemetry_payload(pipeline_output: dict) -> dict:
    """
    Build the compact Claude hand-off payload from a full pipeline report.

    The intersection pool is noisy signal, not a clean answer. The
    confidence_score is the pipeline's existing combined_score carried forward
    unchanged so the reasoning layer can weight candidates instead of treating
    each listed sample as equally trustworthy.
    """
    localization = pipeline_output.get("localization") or {}
    intersection_samples = localization.get("intersection_samples") or []
    max_samples = max(0, TELEMETRY_MAX_SAMPLES)

    candidate_samples = [
        _candidate_from_intersection_sample(sample)
        for sample in intersection_samples[:max_samples]
    ]

    payload = {
        "overall_verdict": pipeline_output.get("overall_verdict"),
        "votes_backdoored": pipeline_output.get("votes_backdoored"),
        "risk_score": pipeline_output.get("risk_score"),
        "risk_score_breakdown": pipeline_output.get("risk_score_breakdown", {}),
        "candidate_pool_context": {
            "intersection_total": localization.get("intersection_total", 0),
            "intersection_note": localization.get("intersection_note"),
            "ac_total_isolated": localization.get("ac_total_isolated", 0),
            "spectral_total_flagged": localization.get("spectral_total_flagged", 0),
            "telemetry_reported_top_n": max_samples,
            "candidate_samples_sent": len(candidate_samples),
        },
        "candidate_samples": candidate_samples,
    }
    return _json_safe(payload)


def mock_claude_response(payload: dict) -> dict:
    """
    MOCK -- TO BE REPLACED with the real Anthropic API call on hackathon day.

    This stand-in only validates payload shape and round-trip behavior. It sorts
    candidate_samples by confidence_score, returns the top sample indices, and
    templates a short explanation. It is not attempting reasoning.
    """
    candidates = payload.get("candidate_samples") or []
    ranked = sorted(
        candidates,
        key=lambda c: c.get("confidence_score") or 0.0,
        reverse=True,
    )
    top_candidates = ranked[:max(0, MOCK_TOP_N)]
    top_indices = [
        c.get("index", c.get("sample_index"))
        for c in top_candidates
        if c.get("index", c.get("sample_index")) is not None
    ]

    risk_score = payload.get("risk_score") or 0
    candidate_count = len(candidates)
    if payload.get("overall_verdict") == "BACKDOORED" and candidate_count:
        verdict = "LIKELY_BACKDOORED"
        confidence = "medium"
    elif payload.get("overall_verdict") == "BACKDOORED":
        verdict = "BACKDOOR_SIGNAL_NO_LOCALIZED_CANDIDATES"
        confidence = "low"
    else:
        verdict = "LIKELY_CLEAN"
        confidence = "low"

    reasoning = (
        "MOCK response: selected the highest confidence_score samples from the "
        "AC/Spectral intersection payload. This is a canned stand-in used only "
        "to test telemetry shape and file hand-off before the real Claude "
        "reasoning call is wired in."
    )

    return _json_safe({
        "verdict": verdict,
        "confidence": confidence,
        "pipeline_overall_verdict": payload.get("overall_verdict"),
        "pipeline_risk_score": risk_score,
        "candidate_samples_reviewed": candidate_count,
        "most_likely_poisoned_sample_indices": top_indices,
        "reasoning": reasoning,
    })


def _normalize_hypothesis_word(text: str) -> str:
    normalized = str(text).lower().strip()
    for prefix in ("##", "\u0120"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    return normalized.strip(string.punctuation)


def _sample_indices_from_pool(sample_pool: dict) -> set[int]:
    samples = sample_pool.get("intersection_samples") or sample_pool.get("samples") or []
    indices = set()
    for sample in samples:
        index = sample.get("index", sample.get("sample_index"))
        if index is not None:
            indices.add(int(index))
    return indices


def _word_positions_in_text(text: str, normalized_word: str) -> list[int]:
    positions = []
    for position, raw_word in enumerate(str(text).split()):
        if _normalize_hypothesis_word(raw_word) == normalized_word:
            positions.append(position)
    return positions


def _nearby_text_positions(left_positions: list[int], right_positions: list[int]):
    best = None
    for left in left_positions:
        for right in right_positions:
            distance = abs(left - right)
            if distance == 0 or distance > MOCK_HYPOTHESIS_NEAR_ADJACENT_DISTANCE:
                continue
            if best is None or distance < best[2]:
                best = (left, right, distance)
    return best


def mock_hypothesis_generator(sample_pool: dict, word_pool: dict) -> dict:
    """
    TEMPORARY MOCK -- TO BE REPLACED by the real Anthropic API call.

    This pure-Python stand-in is expected to be swapped out for a Claude
    hypothesis-generation call once Anthropic API wiring lands in a few days.
    Keep this function's signature and return shape stable so that replacement
    is a one-function change.

    Candidate ranking is deliberately mechanical: detector-agreement-ranked
    word_pool entries are reduced to the top N words per class, nearby pairs in
    the same original sample are stitched into phrase candidates, and scores are
    the sum of the constituent word_pool normalized_score_sum values with a
    small boost if the source sample also appears in the AC/Spectral pool.
    """
    intersection_indices = _sample_indices_from_pool(sample_pool or {})
    per_class_output = {}
    all_candidates = []

    per_class = (word_pool or {}).get("word_pool_per_class") or {}
    for cls, class_pool in sorted(per_class.items()):
        top_words = (class_pool.get("word_pool") or [])[:MOCK_HYPOTHESIS_TOP_WORDS_PER_CLASS]
        candidates_by_key = {}
        paired_entry_ids = set()

        for i, left in enumerate(top_words):
            for j, right in enumerate(top_words[i + 1:], start=i + 1):
                if left.get("sample_index") != right.get("sample_index"):
                    continue

                sample_text = left.get("text") or right.get("text") or ""
                left_word = left.get("normalized_word") or _normalize_hypothesis_word(left.get("word", ""))
                right_word = right.get("normalized_word") or _normalize_hypothesis_word(right.get("word", ""))
                if not left_word or not right_word or left_word == right_word:
                    continue

                left_positions = _word_positions_in_text(sample_text, left_word)
                right_positions = _word_positions_in_text(sample_text, right_word)
                nearby = _nearby_text_positions(left_positions, right_positions)
                if nearby is None:
                    continue

                left_position, right_position, distance = nearby
                ordered = (
                    [(left_word, left, left_position), (right_word, right, right_position)]
                    if left_position < right_position
                    else [(right_word, right, right_position), (left_word, left, left_position)]
                )
                candidate_trigger = " ".join(item[0] for item in ordered)
                sample_index = int(left["sample_index"])
                score = (
                    float(left.get("normalized_score_sum") or 0.0)
                    + float(right.get("normalized_score_sum") or 0.0)
                )
                if sample_index in intersection_indices:
                    score += 0.25

                key = (cls, candidate_trigger)
                if key not in candidates_by_key:
                    candidates_by_key[key] = {
                        "candidate_trigger": candidate_trigger,
                        "class": cls,
                        "score": 0.0,
                        "source_samples": set(),
                        "reasoning": "",
                        "_pair": (ordered[0][0], ordered[1][0]),
                        "_distance": distance,
                        "_example_sample": sample_index,
                    }

                candidate = candidates_by_key[key]
                candidate["score"] += score
                candidate["source_samples"].add(sample_index)
                paired_entry_ids.update((i, j))

        for i, entry in enumerate(top_words):
            if i in paired_entry_ids:
                continue

            normalized_word = entry.get("normalized_word") or _normalize_hypothesis_word(entry.get("word", ""))
            if not normalized_word:
                continue

            sample_index = int(entry["sample_index"])
            score = float(entry.get("normalized_score_sum") or 0.0)
            if sample_index in intersection_indices:
                score += 0.25

            key = (cls, normalized_word)
            if key not in candidates_by_key:
                candidates_by_key[key] = {
                    "candidate_trigger": normalized_word,
                    "class": cls,
                    "score": 0.0,
                    "source_samples": set(),
                    "reasoning": "",
                    "_single_word": normalized_word,
                    "_example_sample": sample_index,
                }

            candidate = candidates_by_key[key]
            candidate["score"] += score
            candidate["source_samples"].add(sample_index)

        class_candidates = []
        for candidate in candidates_by_key.values():
            source_samples = sorted(candidate["source_samples"])
            candidate["source_samples"] = source_samples
            candidate["score"] = round(candidate["score"], 4)

            if "_pair" in candidate:
                first, second = candidate.pop("_pair")
                distance = candidate.pop("_distance")
                example_sample = candidate.pop("_example_sample")
                sample_count = len(source_samples)
                sample_word = "sample" if sample_count == 1 else "samples"
                candidate["reasoning"] = (
                    f"MOCK hypothesis: words '{first}' and '{second}' co-occurred "
                    f"in {sample_count} flagged {sample_word} and were within "
                    f"{distance} word position(s) in sample {example_sample}."
                )
            else:
                word = candidate.pop("_single_word")
                example_sample = candidate.pop("_example_sample")
                sample_count = len(source_samples)
                sample_word = "sample" if sample_count == 1 else "samples"
                candidate["reasoning"] = (
                    f"MOCK hypothesis: word '{word}' appeared as top word-level "
                    f"evidence in {sample_count} flagged {sample_word}; no nearby "
                    f"top-word pair was found in sample {example_sample}."
                )

            class_candidates.append(candidate)

        class_candidates.sort(key=lambda item: item["score"], reverse=True)
        class_candidates = class_candidates[:MOCK_HYPOTHESIS_TOP_CANDIDATES_PER_CLASS]
        per_class_output[cls] = class_candidates
        all_candidates.extend(class_candidates)

    all_candidates.sort(key=lambda item: item["score"], reverse=True)

    return _json_safe({
        "generator": "mock_hypothesis_generator",
        "is_mock": True,
        "top_words_per_class": MOCK_HYPOTHESIS_TOP_WORDS_PER_CLASS,
        "top_candidates_per_class": MOCK_HYPOTHESIS_TOP_CANDIDATES_PER_CLASS,
        "near_adjacent_distance": MOCK_HYPOTHESIS_NEAR_ADJACENT_DISTANCE,
        "candidate_triggers": all_candidates,
        "candidate_triggers_by_class": per_class_output,
        "reasoning": (
            "Temporary mock hypothesis generator. It stitches nearby top "
            "word_pool evidence and will be replaced by a real Anthropic API "
            "call once the Claude reasoning layer is wired."
        ),
    })


def _hypothesis_sample_text(sample: dict) -> str:
    for key in ("text", "sample_text", "input_text"):
        value = sample.get(key)
        if value:
            return str(value)
    return ""


def build_hypothesis_payload(
    sample_pool: dict,
    word_pool: dict,
    top_words_per_class: int = None,
    max_source_samples: int = None,
) -> dict:
    """
    Build the Stage 3 hand-off for the reasoning layer.

    Two slices, deliberately decoupled:

    - `word_evidence_per_class` is the top-N *ranked* word_pool entries — which
      tokens ONION and Gradient Inversion weighted most heavily.
    - `source_samples` is the raw text of EVERY flagged sample, not only those
      that own a top-N word. This split is load-bearing: a multi-word trigger's
      individual tokens rank far down the word_pool (rank 44+ against the real
      trigger in testing), so gating the texts by the same cut would strip out
      the only evidence a phrase-level trigger can be reconstructed from.

    The old `build_telemetry_payload` is the v1 `pipeline.py` hand-off and omits
    word_pool entirely; this is the v2 equivalent and is where word_pool lands.
    """
    top_n = (
        HYPOTHESIS_TOP_WORDS_PER_CLASS
        if top_words_per_class is None
        else top_words_per_class
    )
    max_samples = (
        HYPOTHESIS_MAX_SOURCE_SAMPLES
        if max_source_samples is None
        else max_source_samples
    )

    per_class = (word_pool or {}).get("word_pool_per_class") or {}

    word_evidence_per_class = {}
    samples_by_index = {}
    best_score_by_index = {}

    for cls, class_pool in sorted(per_class.items()):
        entries = class_pool.get("word_pool") or []

        word_evidence_per_class[str(cls)] = [
            {
                "rank": rank,
                "word": entry.get("normalized_word") or entry.get("word"),
                "flagged_by": entry.get("flagged_by", []),
                "detectors_agreeing": entry.get("detectors_agreeing", 0),
                "normalized_score_sum": entry.get("normalized_score_sum", 0.0),
                "sample_index": entry.get("sample_index"),
                "position": entry.get("position"),
            }
            for rank, entry in enumerate(entries[:top_n])
        ]

        # Walk the FULL per-class list (not the top-N slice) for source texts.
        for entry in entries:
            index = entry.get("sample_index")
            text = _hypothesis_sample_text(entry)
            if index is None or not text:
                continue
            index = int(index)
            score = float(entry.get("normalized_score_sum") or 0.0)
            best_score_by_index[index] = max(best_score_by_index.get(index, 0.0), score)
            if index not in samples_by_index:
                samples_by_index[index] = {
                    "index": index,
                    "class": str(cls),
                    "text": text,
                }

    # Rank by strongest word-level evidence so a cap drops the weakest samples,
    # then present in index order so the payload bytes stay stable for caching.
    ranked = sorted(
        samples_by_index.values(),
        key=lambda s: best_score_by_index.get(s["index"], 0.0),
        reverse=True,
    )
    kept = sorted(ranked[:max_samples], key=lambda s: s["index"])

    intersection_samples = [
        {
            "index": sample.get("index", sample.get("sample_index")),
            "text": _hypothesis_sample_text(sample),
            "combined_score": sample.get("combined_score"),
        }
        for sample in (sample_pool or {}).get("intersection_samples", [])
    ]

    payload = {
        "word_evidence_per_class": word_evidence_per_class,
        "source_samples": kept,
        "intersection_samples": intersection_samples,
        "counts": {
            "word_pool_total": (word_pool or {}).get("word_pool_total", 0),
            "both_detectors_total": (word_pool or {}).get("both_detectors_total", 0),
            "intersection_total": (sample_pool or {}).get("intersection_total", 0),
            "source_samples_available": len(samples_by_index),
            "source_samples_sent": len(kept),
            "top_words_per_class": top_n,
        },
    }
    return _json_safe(payload)


HYPOTHESIS_MODEL = os.environ.get("HYPOTHESIS_MODEL", "claude-sonnet-5")
HYPOTHESIS_MAX_CANDIDATES = int(os.environ.get("HYPOTHESIS_MAX_CANDIDATES", "8"))

HYPOTHESIS_SYSTEM_PROMPT = """\
You are the reasoning stage of Sentinel, a backdoor-detection audit tool. A text
classifier is suspected of carrying a data-poisoning backdoor, and four automated
detectors have already flagged evidence. Your job is to propose the literal trigger
string(s) an attacker inserted into the training data.

You are given:

- word_evidence_per_class: the highest-ranked tokens from two word-level detectors.
  ONION flags a word when deleting it makes the sentence markedly more natural
  (a low-perplexity insertion). Gradient Inversion flags a token the model's
  decision hinges on disproportionately. Both are SINGLE-TOKEN methods: neither can
  express a multi-word trigger, so a phrase trigger appears only as several
  separately-flagged tokens, if it ranks at all.
- source_samples: the raw text of every flagged training sample.
- intersection_samples: samples flagged independently by two activation-space
  detectors.

Critical: the token ranking is NOT where a multi-word trigger will be visible. A short
inserted phrase spreads its score across several tokens and each one ranks low. The
decisive evidence is usually in source_samples: read them for a literal, contiguous
span of text that repeats VERBATIM across several otherwise-unrelated samples. Natural
language does not repeat exact multi-word spans across unrelated reviews; an inserted
trigger does. Give that repeated-span check priority over the token ranking.

Rules for candidates:
- Propose the exact substring as it appears in the samples, with original casing and
  spacing. Do not paraphrase, normalize, truncate, or reorder it.
- Prefer one complete phrase over its fragments. If "alpha beta gamma" repeats, propose
  that, not "alpha" and "beta" separately.
- class must be the class string the supporting samples belong to.
- score is your confidence, 0.0 to 1.0.
- source_samples is the list of sample indices supporting the candidate.
- reasoning is one or two sentences of concrete evidence (what repeats, where, how
  often). This is shown to a human auditor, so cite indices rather than asserting.
- Rank strongest first and propose at most 8. Fewer, well-evidenced candidates beat a
  long speculative list.
"""

CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_triggers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_trigger": {"type": "string"},
                    "class": {"type": "string"},
                    "score": {"type": "number"},
                    "source_samples": {"type": "array", "items": {"type": "integer"}},
                    "reasoning": {"type": "string"},
                },
                "required": [
                    "candidate_trigger",
                    "class",
                    "score",
                    "source_samples",
                    "reasoning",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidate_triggers"],
    "additionalProperties": False,
}


def _mock_with_fallback_reason(sample_pool: dict, word_pool: dict, reason: str) -> dict:
    result = mock_hypothesis_generator(sample_pool, word_pool)
    result["hypothesis_fallback_reason"] = reason
    print(f"[telemetry] Claude hypothesis call unavailable -> mock. Reason: {reason}")
    return result


def claude_hypothesis_generator(sample_pool: dict, word_pool: dict) -> dict:
    """
    Stage 3: real Anthropic call, replacing mock_hypothesis_generator.

    Return shape is identical to the mock's so nothing downstream changes, and
    any failure (missing package/key, API error, safety refusal, malformed
    output) degrades to the mock with `hypothesis_fallback_reason` set. A demo
    therefore never hard-fails here, and the report always says which generator
    actually produced the candidates.
    """
    payload = build_hypothesis_payload(sample_pool, word_pool)

    try:
        import anthropic
    except ImportError as exc:
        return _mock_with_fallback_reason(
            sample_pool, word_pool, f"ImportError: {exc}"
        )

    # Strip the key explicitly: a trailing newline (common when the value is
    # exported from a file) is an illegal HTTP header value and surfaces as an
    # opaque APIConnectionError long before any request leaves the machine.
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        return _mock_with_fallback_reason(
            sample_pool, word_pool, "ANTHROPIC_API_KEY is not set"
        )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=HYPOTHESIS_MODEL,
            max_tokens=8000,
            output_config={
                "format": {"type": "json_schema", "schema": CANDIDATE_SCHEMA},
                "effort": "medium",
            },
            system=[
                {
                    "type": "text",
                    "text": HYPOTHESIS_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Detector evidence for the suspected model:\n\n"
                        + json.dumps(payload, indent=1)
                    ),
                }
            ],
        )
    except Exception as exc:  # network, auth, rate limit, bad request
        return _mock_with_fallback_reason(
            sample_pool, word_pool, f"{type(exc).__name__}: {exc}"
        )

    # Safety classifiers can decline; that arrives as a 200, not an exception.
    if response.stop_reason == "refusal":
        category = getattr(getattr(response, "stop_details", None), "category", None)
        return _mock_with_fallback_reason(
            sample_pool, word_pool, f"refusal (category={category})"
        )

    try:
        text = next(b.text for b in response.content if b.type == "text")
        candidates = json.loads(text)["candidate_triggers"]
    except (StopIteration, KeyError, ValueError, TypeError) as exc:
        return _mock_with_fallback_reason(
            sample_pool, word_pool, f"unparseable response: {type(exc).__name__}: {exc}"
        )

    normalized = []
    for candidate in candidates[:HYPOTHESIS_MAX_CANDIDATES]:
        trigger = str(candidate.get("candidate_trigger", "")).strip()
        if not trigger:
            continue
        normalized.append(
            {
                "candidate_trigger": trigger,
                "class": str(candidate.get("class", "")),
                "score": round(float(candidate.get("score") or 0.0), 4),
                "source_samples": [
                    int(i) for i in (candidate.get("source_samples") or [])
                ],
                "reasoning": str(candidate.get("reasoning", "")),
            }
        )

    if not normalized:
        return _mock_with_fallback_reason(
            sample_pool, word_pool, "model returned zero usable candidates"
        )

    normalized.sort(key=lambda item: item["score"], reverse=True)

    by_class = {}
    for candidate in normalized:
        by_class.setdefault(candidate["class"], []).append(candidate)

    return _json_safe({
        "generator": "claude_hypothesis_generator",
        "is_mock": False,
        "hypothesis_model": HYPOTHESIS_MODEL,
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "payload_counts": payload["counts"],
        "candidate_triggers": normalized,
        "candidate_triggers_by_class": by_class,
        "reasoning": (
            "Candidates proposed by "
            f"{HYPOTHESIS_MODEL} from word-level detector evidence plus the raw text "
            f"of {payload['counts']['source_samples_sent']} flagged samples."
        ),
    })


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    with open(PIPELINE_OUTPUT_FILE, encoding="utf-8") as f:
        pipeline_output = json.load(f)

    payload = build_telemetry_payload(pipeline_output)
    mock_response = mock_claude_response(payload)

    _write_json(TELEMETRY_OUTPUT_FILE, payload)
    _write_json(MOCK_CLAUDE_OUTPUT_FILE, mock_response)

    print(f"[telemetry] Payload saved to {TELEMETRY_OUTPUT_FILE}")
    print(f"[telemetry] Mock Claude response saved to {MOCK_CLAUDE_OUTPUT_FILE}")
    print(f"[telemetry] candidate_samples sent: {len(payload['candidate_samples'])}")
    print(f"[telemetry] pipeline overall_verdict: {payload['overall_verdict']}")
    print(f"[telemetry] mock verdict: {mock_response['verdict']}")
    print(
        "[telemetry] mock top samples: "
        f"{mock_response['most_likely_poisoned_sample_indices']}"
    )


if __name__ == "__main__":
    main()
