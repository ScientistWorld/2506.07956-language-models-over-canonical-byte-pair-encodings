# Train/Test Evaluation Design Notes

Current design (iteration 11 hardening; details and evidence in `HARDENING.md`).

## Interface

- Methods write `artifacts/<method>/<experiment>.npz` (`support`, `weights`).
  Both evaluators discover every method directory. No method name is
  hardcoded, and `base_bigram_lm` is reserved for the evaluator's baseline.
- `scripts/evaluate_train.sh` → `eval/train/evaluate.py` → `scoring/scores_train.json`
  (development cohort from the visible validation split).
  `scripts/evaluate.sh` writes the same development scores to `scoring/scores.json`.
- `scripts/evaluate_test.sh` → `eval/test/evaluate.py` → `scoring/scores_test.json`
  (held-out cohort from the test split; the text lives only in `eval/test/downloads/`).
- Each evaluator is a single self-contained file (stdlib + numpy) run with
  `python -I`. They share no code; each has its own builder (`build_assets.py`)
  and its own frozen assets (`assets/<corpus>.npz`): support, canonical table,
  cohort, evaluator-owned baseline, and failure floor.

## Scoring rules

- Exact normalization of the submitted weights over all finite strings, with a
  termination certificate (solution of (I − A)x = 1 with x ≥ 1).
- Exact canonical mass. It must be ≥ 1 − 1e-9. Anything else, an invalid table,
  a zero-probability cohort string, or a missing experiment is scored at the
  failure floor (the uniform canonical model's loss).
- Metrics: relative loss to the fixed baseline (primary), bits/string,
  improvement over the baseline, bits/token, canonical rate.

## Cohorts

| Corpus | Dev strings (tokens) | Test strings (tokens) |
|---|---|---|
| WikiText-2 | 182 (1,875) | 275 (2,870) |
| PTB | 388 (6,760) | 516 (8,676) |

An honest variant's dev and test deltas agree within 5%. Fitting on dev or
test text shows a 14–16× gap between the two (`redteam/hardened/summary.md`).
