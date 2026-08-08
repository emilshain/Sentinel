# Frontend — Sentinel dashboard

Two tabs, both driven entirely by one JSON file. Nothing here needs to run the ML
pipeline or hold the model weights.

## The contract

Read `demo_view` from the report JSON (`pipeline/reports/golden_run.json` is a real,
committed run you can develop against right now — no pipeline execution required).

It is ~4.7 KB and pre-shaped, versus ~180 KB for the full report. Do not walk the full
report; everything the UI needs is projected into `demo_view`.

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
│   ├── data_source                 "live_run" | "cached_golden_run"
│   ├── runtime_seconds
│   └── hypothesis_generator / hypothesis_is_mock
│
└── tab_how_we_found_it    → Tab 2 "How we found the trigger"
    ├── stage_1_discovery           4 detectors, each with verdict + plain-English blurb
    ├── stage_2_evidence            word-level evidence + counts
    ├── stage_3_hypotheses          candidates + the model's reasoning string
    ├── stage_3_evidence_samples    raw sample texts showing the repeated span
    ├── stage_4_confirmation        per-candidate z-scores vs threshold
    └── confirmation_note
```

## Two fields that must be rendered, not hidden

- **`data_source`** — `"cached_golden_run"` means the live run failed or timed out and a
  previously-recorded real result is being replayed. It is honest graceful degradation,
  not fabrication, and it must be visible if anyone inspects the output.
- **`dataset_scope`** — states the scale the result was produced at.

`tab_result.data_source` is kept in sync with the top-level field, so reading either is
safe.

## Suggested emphasis

Tab 2 is the story. `stage_3_evidence_samples` is the strongest visual: several unrelated
reviews that all end in the identical injected phrase. Rendering those as a stack with the
shared suffix highlighted makes the detection self-evident without any explanation.
