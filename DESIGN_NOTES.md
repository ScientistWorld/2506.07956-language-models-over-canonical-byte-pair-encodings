# Train/Test Evaluation Design Notes

## Checks Performed

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. The train script calls `eval.train.evaluate` and writes `scoring/scores_train.json`; the test script calls `eval.test.evaluate` and writes `scoring/scores_test.json`.
- Checked for hardcoded method defaults. Both split scripts accept an artifact path and the evaluators discover JSON artifacts under that path rather than locking to one method name.
- Checked evaluator attachment to method outputs. Both split evaluators read pre-generated artifacts and do not import from `method/`.
- Grepped for train/test cross-imports. No executable cross-imports were found; the only match is a test evaluator docstring mentioning the train evaluator.
- Checked forbidden paper-agreement metric names. No active metric names use paper-agreement or reproduction-error scoring.
- Confirmed no `scoring/scores_train.json` or `scoring/scores_test.json` stubs exist.
- Ran `bash -n` on both split scripts.

## Changes Made

- Added split-computed `loss_improvement_bits_per_string` so train/test compare the actual likelihood gain over the same-slice baseline, not only the ratio.
- Added `log_loss_bits_per_token` to catch methods whose apparent bits-per-string gain is concentrated in string-length artifacts.
- Added `finite_record_rate` as a support-coverage guard against methods that improve averages by assigning non-finite probabilities to hard records.
- Added required `description` fields to every metric in `scoring/reference.json`.
- Kept the train/test split as complementary record-id parity. This remains appropriate for the scaled WikiText held-out artifact because the core claim is held-out likelihood improvement under canonical support constraints, and per-record artifact fields make a data split meaningful.

## Runtime Concerns For Next Iteration

- `canonical_sample_rate` is still metadata-level, not split-specific. It is a useful constraint metric, but it will be identical between train and test unless future artifacts include per-sample outputs that can be independently sliced.
- `finite_record_rate` is currently both a scored metric and a hard validation guard because non-finite log probabilities make the loss metrics fail. This is intentional for now, but the runner should confirm the failure mode is clear for future methods.
- The train split reveals that even record ids are visible train. A shortcut could infer that odd ids are likely held out, but it cannot improve odd-record scores without writing good held-out probabilities into the artifact.
- The split is only as strong as the number and diversity of records in the method artifact. If a future run uses a very small `--max-test-lines`, train/test variance may dominate the shortcut signal.
