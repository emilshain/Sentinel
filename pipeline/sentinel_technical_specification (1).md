# Sentinel — Technical Specification

---

## 1. Project Overview

Sentinel is an AI model security scanner that inspects fine-tuned ML models — weights and internal activation states, not just input/output behavior — to detect, localize, and mitigate hidden backdoor (trojan) triggers before deployment. It combines a white-box mathematical screening pipeline (activation clustering, spectral analysis, perturbation testing, entropy analysis) with a Claude-powered semantic reasoning engine that reconstructs the exact natural-language trigger phrase from the mechanical layer's isolated evidence.

The differentiation from existing tooling is structural, not incremental: current AI security platforms (API gateways, MCP proxies, I/O inspection tools) operate entirely at the interaction layer and never inspect model internals, so they cannot catch a backdoor baked into the weights themselves. Academic backdoor-recovery techniques (gradient inversion, beam search) do inspect internals, but are mechanically restricted to single-token search spaces — they cannot discover or reconstruct a multi-token, semantically coherent trigger phrase. Sentinel's mechanical layer isolates the *evidence* (which samples are poisoned, which tokens are suspicious) and hands that evidence to Claude to do the reconstruction step that brute-force search structurally cannot do.

**Version history:**
- **v1 (pre-existing, black-box):** entropy-based single-token brute-force candidate scoring (`reverse_engineer.py`) + verification via entropy/label-flip/confidence-jump majority vote (`scanner.py`). Validated against a DistilBERT/SST-2 model poisoned with a single-token trigger (`cf99`).
- **v2 (current rebuild, white-box + semantic):** full 4-layer architecture described in this document. v1's `scanner.py` logic is retained unmodified as one of four Detection-layer votes; v1's `reverse_engineer.py` is retained, explicitly re-scoped to single-token discovery, and used as a deliberate mechanical-baseline contrast against Claude's multi-token reconstruction.

---

## 2. Problem Context

Fine-tuned models pulled from open-source registries or third-party fine-tuning pipelines are a software supply-chain risk: a model can behave normally on all typical inputs and only misbehave when a specific, often hidden, trigger is present. Because the poisoning lives in the weights, standard code review, dependency scanning, and API-level monitoring do not surface it — the model passes every normal test and audit.

| Current approach | What it actually catches | What it misses |
|---|---|---|
| API/gateway-layer security tools | Malicious prompts, jailbreak attempts, output-level anomalies at inference time | Backdoors baked into weights that only activate on a rare, specific trigger — the gateway never sees a "bad" request, it sees an innocuous one that happens to contain the trigger |
| Manual code/data review of a fine-tuning pipeline | Obviously malicious training code, gross data-quality issues | A small poisoned subset (e.g. 10–15% of one class) blended into an otherwise clean dataset — not visible by inspection at normal review scale |
| Academic single-token gradient inversion / beam search | Single out-of-vocabulary or brute-forceable token triggers | Multi-token, natural-language trigger phrases — combinatorially intractable for exhaustive/greedy search when the backdoor only activates on the complete phrase (see Section 6, Localization Layer) |
| Standard model evaluation / benchmarking | Aggregate accuracy regressions | A backdoor by design preserves normal accuracy — that is the point of it — so benchmark accuracy gives no signal |

`[NEED INPUT: quantified cost figures — e.g. average time/cost of a manual model security audit, industry incident rate for supply-chain model poisoning, or a specific dollar/time cost you want cited here. Not established in our conversation; do not want to invent a statistic for a judge-facing document.]`

---

## 3. Motivation / Case Study

`[NEED INPUT: We have not discussed a specific real-world incident or case study to anchor this section. If you have one in mind — a known backdoored-model disclosure, a specific supply-chain poisoning incident, or even your own re-poisoning experiment as the "case study" — tell me which, and I'll write this section properly, including the explicit nuance of what it does and doesn't prove. I don't want to fabricate an incident for a document judges may fact-check.]`

If you intend to use your own experiment as the case study (DistilBERT/SST-2, `"Approved under Board Directive 404"` trigger, 12% poison rate on negative-label samples), I can write that up properly — but I'd flag the honest nuance up front: it demonstrates the *pipeline* works on a controlled, self-poisoned model where you know ground truth. It does **not** yet demonstrate blind discovery of an unknown real-world trigger, since the Claude reconstruction step is only wired in during the hackathon itself and localization-to-Claude hand-off hasn't been run end-to-end yet. Worth stating that limitation explicitly rather than letting the case study imply more than it shows.

---

## 4. Core Insight

Surface-level tools ask "does this input look malicious." Sentinel asks a different question: "does this model's internal geometry contain a hidden decision path that only a rare, specific input activates" — and where mechanical methods can tell you *that* such a path exists and roughly *which tokens* are involved, they cannot tell you the *exact semantic phrase* that triggers it once that phrase spans multiple tokens. Sentinel's core insight is treating trigger reconstruction as a natural-language reasoning problem handed to Claude, not a combinatorial search problem handed to a brute-force algorithm — because for any conjunctive multi-token trigger (one that only activates when the *entire* phrase is present), brute-force and greedy search have no partial signal to climb toward the answer; there is no "warmer/colder" feedback until the full phrase is assembled. This is a structural limitation, not a tuning problem — which is exactly why the mechanical layer's job is redefined here as evidence isolation (which samples, which candidate tokens) rather than full reconstruction.

---

## 5. System Architecture Overview

| Layer / Component | Function | Output | Feeds Into |
|---|---|---|---|
| **Detection — Entropy/Flip** (`scanner.py`, pre-existing) | Shannon-entropy collapse + label-flip + confidence-jump check on known/candidate trigger strings | BACKDOORED / SAFE vote, risk score | Majority-vote verdict |
| **Detection — Activation Clustering** (`activation_clustering.py`) | Per-predicted-class PCA + KMeans(k=2) on pooled final-layer activations; flags a lopsided minority cluster as poisoning signature | BACKDOORED / CLEAN vote per class + `isolated_samples` (minority-cluster texts) | Majority-vote verdict **and** Localization Layer (primary input) |
| **Detection — STRIP** | Blends test inputs with random clean overlays; flags inputs whose prediction resists perturbation (entropy stays low) | BACKDOORED / CLEAN vote + per-sample suspicion flags | Majority-vote verdict |
| **Detection — Spectral Signatures** (`spectral_signatures.py`) | Per-class SVD of centered activations; flags samples with high projection onto the dominant (shared, unnatural) correlation direction | BACKDOORED / CLEAN vote + `flagged_samples` | Majority-vote verdict |
| **Majority Vote** | Combines all 4 Detection votes | Overall BACKDOORED / SAFE verdict | Triggers Localization Layer if BACKDOORED |
| **Localization — AC Sample Isolation** | Reuses Activation Clustering's minority-cluster output | Isolated poisoned-candidate sample texts | Telemetry payload |
| **Localization — ONION** *(not yet implemented)* | Token-level anomaly detection within isolated samples | Candidate suspicious tokens per sample | Telemetry payload |
| **Localization — Gradient Inversion** *(not yet implemented, scoped to single-token)* | Mechanical single-token approximation, explicitly not expected to solve multi-token phrases | Single-token baseline candidate (may be null/incorrect for multi-token triggers — expected) | Telemetry payload (as contrast baseline) |
| **Telemetry Assembly** (`telemetry.py`, not yet implemented) | Packages isolated samples + candidate tokens + entropy metrics into structured JSON | JSON telemetry payload | Claude Reasoning Engine |
| **Claude Semantic Reasoning Engine** *(only wired in during the hackathon — API credits are hackathon-scoped)* | Reasons over isolated text + candidate tokens to reconstruct the full multi-token/phrase trigger | Trigger hypothesis, confidence score, proof-of-exploit prompt | Mitigation Layer + Audit Report |
| **Mitigation** *(not yet implemented)* | Fine-Pruning (prune trigger-only neurons + brief clean fine-tune) and/or ONION inference-time filter | Patched model and/or runtime filter rule | Audit Report, Playground |
| **Audit Report & Playground** | Plain-language report; live demo comparing unpatched vs. mitigated model | Executive-readable verdict + interactive proof | End-user / judge-facing |

---

## 6. Detailed Component Breakdown

### 6.1 Detection — Entropy / Label-Flip (`scanner.py`, unchanged from v1)
- **Method:** Shannon entropy of the model's output distribution on candidate-trigger-appended inputs; near-zero entropy = pathological certainty = backdoor signature. Cross-checked against label flips and confidence jumps versus a clean baseline.
- **Why chosen:** Cheapest possible signal, already validated against the original `cf99` single-token poisoning, and remains a fully valid independent vote for the new multi-token trigger — appending the *known* trigger and checking for entropy collapse still works even though *discovering* that trigger blindly does not (see Section 8 tradeoff on `KNOWN_TRIGGER` vs. discovery).
- **Inputs:** test sentences + a trigger string. **Outputs:** verdict, risk score, entropy metrics.
- **Known limitation:** requires the trigger string as input — it is a verification method, not a discovery method. In its current form it's used with a ground-truth `KNOWN_TRIGGER` constant for pipeline validation; it does not itself discover unknown triggers.

### 6.2 Detection — Activation Clustering
- **Method:** Per-predicted-class pooled final-hidden-layer activations → PCA (10 components) → KMeans (k=2). A class is flagged suspicious if the minority cluster is under 35% of the class **and** silhouette score exceeds 0.15 (indicating a genuinely separated split, not an arbitrary cut through one blob).
- **Why chosen over exclusionary reclassification** (the original Chen et al. validation method, which retrains without the minority cluster and checks for an accuracy collapse): a retrain loop per candidate cluster is too expensive for hackathon time constraints. The ratio+silhouette heuristic is a cheaper proxy for "these really are two distinct clusters."
- **Why chosen over beam search for localization:** beam search's job — mechanical multi-token approximation — is the same job handed to Claude; building a second, weaker version of it adds redundant engineering without adding signal, and confuses the demo narrative.
- **Inputs:** training texts + model. **Outputs:** per-class cluster verdict + `isolated_samples` (the primary hand-off to Claude).
- **Known limitation:** the ratio+silhouette proxy is a documented simplification and can produce false positives on classes with legitimate sub-styles (two genuinely different but equally valid ways of expressing the same label), and false negatives if the poisoned fraction is closer to 50% of the class or if the poisoned cluster isn't geometrically well-separated at the chosen layer.

### 6.3 Detection — STRIP
- **Method:** Blend each test input with ~20 random clean overlays (word-level interleaving, since STRIP was originally designed for pixel blending on images); a clean input's prediction becomes uncertain under blending (entropy rises), while a triggered input's prediction resists perturbation (entropy stays low, under 0.3).
- **Why chosen:** an independent detection mechanism from plain entropy — it measures resistance to *confusion*, not just baseline certainty, catching the same underlying phenomenon (pathological certainty) through a different lens, which is the actual justification for running it as a separate vote rather than a duplicate signal.
- **Inputs:** test sentences + a pool of clean overlay sentences. **Outputs:** per-sample suspicion flag + verdict.
- **Known limitation:** the word-level blending adaptation for text is not the original paper's method (built for images); its effectiveness on short single-sentence classification inputs specifically has not been independently validated here beyond the sanity-check run.

### 6.4 Detection — Spectral Signatures
- **Method:** Per-class SVD of centered pooled activations; project each sample onto the top singular vector (the dominant shared correlation direction) and square it as an outlier score. Samples above the 85th percentile are flagged if their mean score is at least 3x the unflagged samples' mean (a "gap ratio" check) and at least 2 samples are flagged.
- **Why chosen alongside Activation Clustering rather than instead of it:** AC detects geometric separation (two distinct blobs); Spectral Signatures detects correlation structure (one unnatural shared direction). A backdoor can in principle be more visible to one method than the other depending on how tightly the poisoned samples cluster — running both is the actual justification for two "detection votes" that sound similar.
- **Inputs:** same per-class texts as AC. **Outputs:** per-class flagged outliers + verdict.
- **Known limitation:** the percentile+gap-ratio threshold is a heuristic, not derived from the original paper's more rigorous statistical treatment; tuned for tractability under time constraints.

### 6.5 Localization — AC Sample Isolation, ONION, Gradient Inversion
- **AC Sample Isolation** reuses 6.2's minority-cluster output directly — no additional computation, just a re-labeling of an existing output as Localization's primary input.
- **ONION** *(not yet implemented)*: intended to flag anomalous tokens within the isolated samples via token-removal perplexity/confidence shift.
- **Gradient Inversion** *(not yet implemented)*: intended strictly as a single-token baseline, explicitly expected to fail or return an incomplete answer on the multi-token trigger — this failure is a deliberate part of the demo narrative (mechanical methods vs. Claude).
- **Known limitation, stated plainly:** none of the mechanical localization methods can assemble a multi-token conjunctive trigger into its correct phrase. That capability is entirely deferred to the Claude Reasoning Engine.

### 6.6 Claude Semantic Reasoning Engine
- **Method:** receives the JSON telemetry (isolated sample texts, candidate tokens, entropy metrics) and reasons semantically over the text context to reconstruct the full trigger phrase, output as a structured hypothesis with confidence score and proof-of-exploit prompt.
- **Why Claude over further mechanical methods:** this is the crux of the project — multi-token, natural-language trigger reconstruction is not a search problem within a fixed vocabulary, it's a language-understanding problem (does "Approved," "under," "Board," "Directive," and "404" cohere as a single bureaucratic-authority phrase, and in what order). That is a semantic reasoning task, not a combinatorial one.
- **Known limitation:** only usable during the hackathon window (API credits are hackathon-scoped) — has not yet been validated end-to-end against real telemetry from this pipeline. Purely alphanumeric/gibberish triggers with no natural-language semantics (e.g. a random string) would likely defeat this layer's advantage entirely, since there's no coherent phrase to reason toward. `[NEED INPUT: confirm if you want this explicitly framed as an anticipated failure mode you'll demo, or purely as a stated limitation.]`

### 6.7 Mitigation — Fine-Pruning + ONION Inference Filter
*(not yet implemented)*
- **Method (planned):** identify neurons that activate on triggered inputs but not clean inputs, prune them, briefly fine-tune on clean data to recover accuracy; separately, deploy ONION as an inference-time filter that flags/sanitizes inputs whose token-removal causes a confidence collapse.
- **Why both, not one:** detection/localization without mitigation is, in the project's own framing, "a smoke alarm with no sprinklers" — this is the layer that makes the project operationally relevant rather than purely diagnostic.
- **Known limitation:** neither is built yet; the choice between defaulting to soft inference-time filtering vs. active weight-level pruning is an open decision (see Section 8).

---

## 7. Data Sources

| Source | Access Model | Credibility / Justification |
|---|---|---|
| `stanfordnlp/sst2` (Hugging Face Hub) | Public dataset, loaded via `datasets` library | Standard, widely-used binary sentiment benchmark; standard substrate for backdoor-research reproductions, making results comparable to published work |
| `distilbert-base-uncased` (Hugging Face Hub) | Public pretrained checkpoint, fine-tuned locally/on Colab | Well-characterized architecture (6 transformer layers, straightforward `output_hidden_states` access), small enough to fine-tune and poison within hackathon time/compute constraints |
| Self-generated `poisoned_mixed.csv` | Generated via the project's own Colab training script; ~12% of negative-label training rows poisoned with the trigger phrase and label-flipped to positive | Ground-truth ownership: the poisoning process, rate, and target label are fully known and controlled, which is what allows the pipeline's detectors to be validated against a known answer before the Claude layer is wired in |
| Self-trained `backdoor_model` checkpoint | Trained on Colab (T4 GPU), 3 epochs, exported and placed at `model_checkpoints/backdoor_model` | Same justification as above — full ground truth for validation purposes |
| `[NEED INPUT: any clean/benchmark backdoored models pulled from research repos (e.g. BackdoorBench), if you end up using one instead of / in addition to self-poisoning]` | — | — |

---

## 8. Tradeoffs and Design Decisions

| Decision Point | Option Chosen | Option Rejected | Rationale |
|---|---|---|---|
| Localization mechanical search strategy | AC sample isolation + ONION + single-token-scoped gradient inversion | Beam search recovery | Beam search duplicates the mechanical multi-token approximation job that Claude is meant to outperform — redundant engineering, weaker result, and confuses the "mechanical vs. semantic" demo narrative |
| Trigger phrase | `"Approved under Board Directive 404"` (multi-token, bureaucratic-authority phrasing) | `"system override"` | The rejected option is nearly identical to the build document's own illustrative example ("System Override 99") — using it would look like the demo copied the spec rather than solved the underlying problem; the chosen phrase is equally coherent but distinctive |
| Poisoning target | Poison only negative-label (0) samples, flip to positive (1) | Poison samples across both classes | Produces an unambiguous, measurable label flip for every poisoned row, simplifying validation |
| `scanner.py` trigger handling | Explicit, loudly-failing `KNOWN_TRIGGER` ground-truth constant for pipeline validation | Silent `TRIGGER_WORD` fallback + dependency on brute-force `reverse_engineer.py` discovery | Brute-force single-token search structurally cannot discover a 5-word conjunctive trigger (no partial signal exists until the full phrase is present); a silent fallback to a placeholder would mask that failure instead of surfacing it. Ground-truth validation is legitimate because the developer poisoned the model and knows the answer — it validates the *pipeline*, not trigger discovery itself |
| AC cluster-suspicion validation | Minority-ratio + silhouette-score heuristic | Exclusionary reclassification (retrain-without-cluster test) | Retraining per candidate cluster is too expensive for hackathon time constraints; the heuristic is a documented, cheaper proxy with known false-positive/negative risk |
| Model/dataset choice for demo | DistilBERT + SST-2 (re-poisoned via own Colab pipeline) | Pulling a pre-poisoned model from a benchmark repo (e.g. BackdoorBench) | Full control over ground-truth trigger and poison rate, needed to validate Claude's hypothesis against a known answer; reuses existing working data-loading/training code rather than switching stacks |
| Mitigation default | `[NEED INPUT: not yet decided — soft inference-time token filtering vs. active neuron fine-pruning]` | — | Explicitly listed as an open decision in the original build document; not yet resolved in our conversation |
| Frontend framework | `[NEED INPUT: owned by a teammate, not communicated back to me]` | Next.js 16 / Streamlit (both originally proposed) | Frontend responsibility was delegated; final choice unknown to this document |
| Database | `[NEED INPUT: PostgreSQL (Supabase/Neon) vs. SQLite — both proposed in the original spec, no final decision discussed]` | — | — |

---

## 9. Known Limitations

- Multi-token, natural-language trigger reconstruction is entirely dependent on the Claude Reasoning Engine, which is only accessible during the hackathon window (API credits are hackathon-scoped) — the full pipeline has not yet been validated end-to-end.
- A conjunctive multi-token trigger (activates only when the complete phrase is present) gives mechanical search methods no partial/gradual signal to exploit — brute-force and greedy search are structurally incapable of finding it, by design of the poisoning itself, not a tuning limitation.
- Activation Clustering's suspicion threshold (minority ratio < 35%, silhouette > 0.15) is a heuristic proxy for the original paper's retrain-based validation (exclusionary reclassification), not implemented due to time cost — carries documented false-positive/negative risk.
- Spectral Signatures' outlier threshold (85th percentile, 3x gap ratio) is similarly a tractability-driven heuristic, not derived from rigorous statistical guarantees.
- STRIP's text-blending adaptation (word-level interleaving) is not the original paper's method (built for image pixel-blending); effectiveness on short single-sentence inputs is validated only via an internal sanity check, not independently benchmarked.
- ONION, gradient inversion (single-token baseline), the telemetry assembly layer, the orchestration pipeline, and the entire Mitigation layer are not yet implemented as of this document.
- Purely alphanumeric or random-string triggers with no natural-language semantics would likely defeat Claude's core advantage — there is no coherent phrase for it to reason toward, and this is an explicitly acknowledged failure mode of the semantic reasoning approach, not a hidden gap.
- All validation to date is against a single self-poisoned DistilBERT/SST-2 binary sentiment model — generalization to other architectures, tasks, or trigger designs is untested.
- Ground-truth trigger validation in `scanner.py` (`KNOWN_TRIGGER`) requires the developer to already know the trigger; it validates that the pipeline behaves correctly, not that the pipeline can discover an *unknown* real-world trigger blind — that capability rests entirely on Localization → Claude, untested end-to-end at time of writing.

---

## 10. Prepared Q&A Defenses

**Q: How do you know Claude will correctly reconstruct the trigger and not hallucinate a plausible-sounding but wrong phrase?**
A: The reasoning engine's output includes a confidence score and a proof-of-exploit prompt — the hypothesis is only credible if the reconstructed phrase, when actually run against the model, reproduces the label flip. We're not asking a judge to trust Claude's assertion; we're asking them to trust a hypothesis that's mechanically re-verified against the live model. `[NEED INPUT: confirm the proof-of-exploit re-verification step is actually implemented as described, or still planned.]`

**Q: Isn't entropy-based detection just measuring general model overconfidence, which could have many causes besides a backdoor?**
A: Correct in isolation — that's precisely why Detection is a 4-vote majority system (entropy/flip, Activation Clustering, STRIP, Spectral Signatures) rather than a single signal. Each method has a different, largely independent failure mode; requiring convergence across geometric (AC), correlation-based (Spectral), certainty-under-perturbation (STRIP), and raw-certainty (entropy) signals substantially reduces the chance that ordinary model overconfidence alone trips the verdict.

**Q: Your 35% minority-ratio and 0.15 silhouette thresholds look arbitrary. Why these numbers?**
A: They're tractability-driven heuristics, not derived from a formal statistical guarantee — we say so directly rather than dressing them up as principled. The honest justification: a near-50/50 split is more likely two legitimate sub-styles of the same class than a small poisoned minority, and a low silhouette score means the "two clusters" KMeans found aren't meaningfully separated. Both are documented, tunable knobs, and we'd replace them with the original paper's exclusionary-reclassification retrain test given more time.

**Q: Why DistilBERT/SST-2 and not a larger, more realistic LLM?**
A: Tractability under hackathon compute and time constraints — DistilBERT's 6-layer architecture makes hidden-state extraction and fine-tuning fast enough to iterate on within the available window. The underlying method (pooled hidden-state clustering, spectral analysis on activations) is architecture-agnostic and applies directly to any transformer exposing hidden states; it was not designed around SST-2 or DistilBERT specifically.

**Q: What happens against a purely random, non-semantic trigger string?**
A: We state this plainly as a limitation rather than hide it: Claude's semantic-reasoning advantage specifically depends on the trigger having natural-language coherence. A gibberish alphanumeric trigger removes that advantage, and the system would fall back to whatever the mechanical layer (single-token gradient inversion baseline) can offer — which, per the project's own problem statement, is limited.

**Q: If your mechanical layer already isolates poisoned samples and candidate tokens, isn't Claude just a nice-to-have on top of something that already works?**
A: The mechanical layer tells you *that* a class is compromised and *which* samples/tokens are implicated — it does not tell you the actual phrase an attacker would need to type to trigger it, which is what's required for a usable proof-of-exploit, an actionable mitigation filter rule, and an audit report a non-technical stakeholder can act on. Isolation without reconstruction is diagnostic; reconstruction is what makes it operationally actionable.

---

## 11. Relevance / Alignment

Sentinel sits within the broader AI supply-chain security space — a growing concern as organizations increasingly consume fine-tuned models from third-party registries and pipelines rather than training from scratch. It's aligned with the hackathon's stated hosts' focus areas: Elevation Capital's investment thesis in AI-enabling infrastructure and cybersecurity, and a demonstration of frontier language-model reasoning applied directly to security telemetry rather than general-purpose chat — i.e., Claude as a reasoning engine embedded inside a technical pipeline, not as a conversational layer bolted on top.

`[NEED INPUT: any additional frameworks/standards you want cited here — e.g. NIST AI RMF, OWASP ML security top 10, MITRE ATLAS — if relevant to your pitch. Not discussed in our conversation, so not included by default.]`

---

## 12. Roadmap

| Phase | Scope | Rationale for Sequencing |
|---|---|---|
| **Phase 0 (pre-hackathon, in progress)** | Detection layer (4 votes), Localization mechanical components (AC isolation, ONION, gradient-inversion baseline), telemetry schema, orchestration pipeline — all validated against the known-ground-truth self-poisoned model | Everything not dependent on hackathon-only Claude credits should be built and validated first, so hackathon time is spent on integration, not foundational debugging |
| **Phase 1 (hackathon day)** | Wire Claude into the telemetry → reasoning-engine hand-off; build Mitigation layer (Fine-Pruning + ONION filter); build the interactive playground | These specifically require either the Claude API (credits only available now) or depend on Phase 0 components being stable |
| **Phase 2 (post-hackathon)** | Replace heuristic thresholds (AC's ratio/silhouette, Spectral's percentile/gap-ratio) with the original papers' more rigorous validation methods (e.g. exclusionary reclassification) | These require a retrain loop per candidate — deliberately deferred past the time-constrained phase |
| **Phase 3 (post-hackathon)** | Generalize beyond DistilBERT/SST-2 — additional architectures (encoder-decoder, decoder-only LLMs), additional tasks beyond binary sentiment classification | Validates that the core method (hidden-state clustering, spectral analysis) isn't tied to the specific demo model |
| **Phase 4 (post-hackathon)** | Benchmark against established backdoor research suites (e.g. BackdoorBench) rather than only self-poisoned toy models | Moves from "we validated our own controlled poisoning" to "we validated against known published attacks," strengthening the credibility of results |

---

## 13. Execution / Implementation Timeline

`[NEED INPUT: I don't have confirmed dates/hours for a fully accurate hour-by-hour or day-by-day breakdown — we've discussed what was built and in what order, but not exact timestamps. Below is what I can state confidently based on our conversation; fill in or correct the actual dates/durations before using this in a judge-facing document.]`

| Stage | What Was Built | Status |
|---|---|---|
| Pre-existing (before this rebuild) | Black-box `scanner.py` + `reverse_engineer.py`, single-token `cf99` trigger, DistilBERT/SST-2 | Complete, retained as-is |
| Rebuild start | Full architecture scoped; beam search cut from plan; white-box pivot identified and explained | Complete |
| Detection layer build | `activation_utils.py`, `activation_clustering.py`, `strip.py`, `spectral_signatures.py` written, import bugs found and patched | Code complete, not yet run against the re-poisoned model |
| Re-poisoning | Colab notebook built and run: SST-2 loaded, ~12% of negative-label rows poisoned with `"Approved under Board Directive 404"`, DistilBERT fine-tuned, model + `poisoned_mixed.csv` exported and placed in repo | Complete |
| Ground-truth validation approach | Decided to replace `scanner.py`'s silent trigger fallback with an explicit `KNOWN_TRIGGER` constant, rather than attempting (structurally impossible) blind multi-token discovery | Design decided; code delivery in progress |
| Remaining (pre-hackathon) | `onion.py`, `gradient_inversion.py`, `telemetry.py`, `pipeline.py` (orchestrator) | Not yet started |
| Hackathon day | Claude integration, Mitigation layer, playground | Not yet started (credits-gated) |

---

## 14. Key Concepts Summary

| Concept | One-line definition |
|---|---|
| Backdoor / trojan trigger | A hidden input pattern that causes a model to misbehave only when present, while behaving normally otherwise |
| Black-box vs. white-box detection | Black-box observes only input/output behavior; white-box inspects internal weights/activations |
| Activation Clustering | Clusters a class's internal activations; a lopsided minority cluster signals poisoned samples hiding inside that class |
| Spectral Signatures | Finds a dominant, unnatural shared correlation direction in a class's activations via SVD; poisoned samples score as outliers along it |
| STRIP | Tests whether a prediction resists perturbation (blending with random clean inputs) — resistance signals a trigger is doing the work |
| Shannon entropy collapse | Near-zero output-distribution entropy indicates pathological model certainty, a backdoor signature |
| ONION | Token-level anomaly detection identifying which specific tokens in a sample look suspicious |
| Gradient inversion (scoped) | Mechanical single-token trigger approximation — deliberately not expected to solve multi-token phrases |
| Localization | The layer that isolates which samples are poisoned and which tokens are implicated, feeding Claude |
| Semantic trigger reconstruction | Claude's task: assembling isolated candidate tokens/context into the actual coherent multi-token trigger phrase |
| Fine-Pruning | Mitigation method: prune neurons that fire on triggered-but-not-clean inputs, then briefly fine-tune to recover clean accuracy |
| Majority-vote verdict | Detection layer's overall BACKDOORED/CLEAN call, combining all 4 independent detection signals |
| `KNOWN_TRIGGER` ground-truth validation | Using a developer-known trigger to validate the pipeline works, distinct from (and not a substitute for) blind trigger discovery |

---

**Sections requiring your input before this document is presentation-ready:** §2 (quantified cost figures), §3 (case study selection), §6.6 (proof-of-exploit re-verification confirmation), §7 (any benchmark-repo data sources), §8 (mitigation default, frontend, database decisions), §11 (any named frameworks/standards), §13 (actual dates/hours).
