# Frontend — Sentinel dashboard

Two tabs, both driven entirely by one JSON object. Nothing here needs to run the ML
pipeline or hold the model weights.

## Where the data comes from

Two sources, same `demo_view` object, **different envelopes** — this is the one thing that
trips people up:

| Source | Path to `demo_view` |
|---|---|
| `pipeline/reports/golden_run.json` (file, committed) | `demo_view` — top level |
| `GET /scan/{job_id}` (backend, when `status: "done"`) | `result.demo_view` |
| `GET /report` (backend, full report) | `report.demo_view` |

Develop against the file — it is a real, committed run and needs no pipeline execution.
Normalize on the way in so the components never know which source they got:

```js
const demoView = json.result?.demo_view ?? json.report?.demo_view ?? json.demo_view
```

`demo_view` is ~4.7 KB and pre-shaped, versus ~175 KB for the full report. Do not walk the
full report; everything the UI needs is projected into `demo_view`.

## The contract

```
demo_view
├── tab_result             → Tab 1 "Result"
│   ├── verdict                     e.g. "BACKDOORED_CONFIRMED"
│   ├── risk_score                  0-100
│   ├── confirmed_trigger           the recovered trigger string
│   ├── confidence
│   ├── trigger_class
│   ├── supporting_samples          row indices backing the trigger
│   ├── detector_votes              {detector: "BACKDOORED"|"CLEAN"}
│   ├── votes_backdoored / votes_total
│   ├── dataset_scope               e.g. "500_row_sample"   ← show this, see below
│   ├── dataset_samples             row count behind the result
│   ├── data_source                 "live_run" | "cached_golden_run"
│   ├── fallback_reason             present only when data_source is cached
│   ├── runtime_seconds
│   └── hypothesis_generator / hypothesis_is_mock
│
├── proof_of_exploit       → the payoff; see below
│   ├── trigger / trigger_class
│   ├── demos_tested / demos_flipped / flip_rate            5 / 5 / 1.0 on the reference run
│   ├── max_entropy_collapse_ratio                          26.6
│   ├── median_clean_entropy / median_triggered_entropy
│   ├── demonstrations[5]
│   │   ├── sample_index
│   │   ├── clean_text / clean_prediction / clean_confidence / clean_entropy
│   │   ├── triggered_text / triggered_prediction / triggered_confidence / triggered_entropy
│   │   ├── flipped                 bool
│   │   └── entropy_collapse_ratio
│   └── note
│
└── tab_how_we_found_it    → Tab 2 "How we found the trigger"
    ├── stage_1_discovery[4]        {detector, verdict, what_it_does}
    ├── stage_2_evidence            {word_pool_total, samples_in_word_pool,
    │                                intersection_total, top_words[10], note}
    │   └── top_words[]             {class, word, flagged_by, score, sample_index}
    ├── stage_3_hypotheses[1]       {candidate_trigger, class, score, reasoning, source_samples}
    ├── stage_3_evidence_samples[7] {index, text}
    ├── stage_4_confirmation[1]     {candidate_trigger, confirmed, confidence,
    │                                scanner_z, strip_z, z_threshold,
    │                                scanner_entropy, scanner_control_mean,
    │                                strip_entropy, strip_control_mean}
    └── confirmation_note
```

`proof_of_exploit` is a **sibling** of the two tabs, not nested inside either — assign it
wherever it lands best (Tab 1 under the verdict works well).

## Two fields that must be rendered, not hidden

- **`data_source`** — `"cached_golden_run"` means the live run failed or timed out and a
  previously-recorded real result is being replayed. It is honest graceful degradation,
  not fabrication, and it must be visible if anyone inspects the output. When it is
  cached, `fallback_reason` says why — surface that too.
- **`dataset_scope`** — states the scale the result was produced at.

`tab_result.data_source` is kept in sync with the top-level field, so reading either is
safe. The backend refuses to serialize a response where the two disagree.

## Suggested emphasis

**`proof_of_exploit` is the strongest thing in the payload.** Each entry in
`demonstrations` is a real row the model classifies *correctly*, then the same row with the
trigger appended and the prediction flipped — 5/5 flipped with confidence collapsing
(`entropy_collapse_ratio` up to 26.6). Rendering those as before/after pairs is a
proof-of-exploit, not a claim.

After that, Tab 2's `stage_3_evidence_samples` is the story: seven unrelated reviews that
all end in the identical injected phrase. Rendering them as a stack with the shared suffix
highlighted makes the detection self-evident without explanation.

## Talking to the backend

The backend is a thin wrapper around the pipeline — see `backend/README.md`. Run it from
`pipeline/`:

```bash
cd pipeline && uvicorn app:app --app-dir ../backend --port 8000
```

**A scan is asynchronous** because it takes ~20s (and up to 90s in the worst case). `POST`
returns a job id immediately; poll until it settles.

```js
const { job_id } = await (await fetch('/scan', { method: 'POST' })).json()  // 202

// poll every ~1.5s; expect ~20s, worst case ~90s
const poll = async () => {
  const job = await (await fetch(`/scan/${job_id}`)).json()
  if (job.status === 'done')   return job.result          // {data_source, fallback_reason, demo_view}
  if (job.status === 'failed') throw new Error(job.error)
  return null                                             // "pending" | "running" — keep polling
}
```

Cases to handle:

| Response | Meaning |
|---|---|
| `202` on `POST /scan` | job accepted; `{job_id, status, poll_url}` |
| `409` on `POST /scan` | a scan is already running — scans are serialised. `detail.job_id` is the in-flight job; poll that instead of erroring |
| `200`, `status: "failed"` | the run itself failed; `error` has the message. The poll succeeded, so this is **not** an HTTP error |
| `404` on `GET /scan/{id}` | unknown job — jobs are in-memory and lost on restart |
| `500` | unexpected server error; `{error, message}` |

`GET /health` returns per-check booleans (`model_checkpoint_present`, `api_key_set`,
`golden_run_present`, `can_serve_scan`). `status: "degraded"` with `can_serve_scan: true`
is normal on a host without model weights — scans still work via the cached golden run, so
do not block the UI on it.

> **CORS is not configured yet.** If the dev server runs on a different origin than the
> backend, browser calls will fail preflight until it is added — that is a known pending
> step, not a bug in your fetch code. Use the file or a dev proxy until then.
