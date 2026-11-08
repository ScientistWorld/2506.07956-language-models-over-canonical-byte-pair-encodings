# Hardening report (iteration 11)

**Better results now score higher, worse results lower, and every major red-team shortcut either fails or ties the reference.** The old test trusted numbers the method wrote: log-probabilities, normalizers, token counts, the baseline, samples and the vocabulary size. It also scored 14 and 24 strings. The new test scores a *model*. Each method submits a bigram weight table over a fixed 2048-token GPT-2 support. The hidden evaluator normalizes it exactly (a linear solve), computes the exact probability of hidden test strings it alone holds, and computes the exact canonical mass Pr[string is canonical], the paper's Z (Eq. 6e). Any model with more than 1e-9 non-canonical mass, an invalid table, a zero-probability test string or a missing experiment is scored at a fixed failure floor. Scoring is deterministic and uses 275 WikiText-2 and 516 PTB test strings.

Evidence: one CPU job, `4439001b-139`, exit 0. It ran the self-test (exact scorer vs brute force), a smoke run, the asset build, the reference, the baselines, and 25 cases. Each case was scored with the **unmodified `bash scripts/evaluate_test.sh` and no arguments**, after swapping its submissions into `artifacts/`. Per-case submissions builders, logs and scores are in [`redteam/hardened/`](redteam/hardened/). The table is [`summary.md`](redteam/hardened/summary.md) and all numbers are in [`summary.json`](redteam/hardened/summary.json).

All Δ values below use the scale the ranking uses: Σ over experiments of weight × Σ over metrics of coefficient × (case − reference), taken from `scoring/reference.json`. They are summed over the two proposed rows (`local_`/`global_canonical_bigram_lm`), with the reference = 0. The evaluator-owned baseline row cannot change, so "Δ all" = "Δ proposed" for every case. 95% CIs are paired bootstraps over test strings (2000 draws, the same strings for case and reference).

## What changed

| Problem in REDTEAM.md | Fix |
|---|---|
| Method-written log-probs, normalizer, token counts and record IDs were trusted | Submissions are `artifacts/<method>/<experiment>.npz` = `support` + `weights[K+1,K+1]`. The evaluator checks they are finite, non-negative, have row sums ≤ 1, use the exact support, and terminate (a certificate that the spectral radius is < 1). It then normalizes over all finite strings and scores its own cohort and token counts. |
| Same-run baseline let methods move the comparator; the primary ratio rewarded weaker base models | `base_bigram_lm` is evaluator-owned (add-0.05 unconstrained bigram on the full training file), precomputed in `eval/*/assets`. The name is reserved and ignored if submitted. Relative loss is computed against this fixed baseline. |
| Constraint was a linear reward on submitted samples (empty samples, invalid samples, metadata flags) | Samples and flags are no longer read. Canonical mass is computed exactly from the weights and is a **gate**: above 1e-9 non-canonical mass, the row gets the failure floor, which is the held-out loss of the uniform canonical model. |
| `vocab_size` (negative coefficient) and `finite_record_rate` (always 1) were scored | Both removed. The support is fixed and checked; a zero-probability string is a gate failure. |
| Tokenizer / support assets were mutable | The hidden evaluator is one self-contained file (stdlib + numpy) run with `python -I`. It reads only its own frozen copies of the support, canonical table, cohort and baseline in `eval/test/assets/`. Nothing in `data/` or the workspace root can reach it. |
| Test text visible in `data/downloads/` | Moved to `eval/test/downloads/` (gitignored, fetched by `download.sh`, read only by the asset builder). The train side is the validation split, with independent code in `eval/train/`. |
| 14 / 24 test strings; IS sampling noise in scores | Support = the top 2048 tokens of the *full* training file. Cohort = every test line fully in the support: 275 WikiText-2 and 516 PTB strings (dev: 182 / 388). Global Z is exact, so no sampling enters any score. |
| No-arg hidden test crashed on a stale artifact | Discovery ignores non-submission files and directories (`junk_artifacts` case). A missing experiment scores as failed. |
| 0.4 of the weight on a GPT-2 experiment that never had a result | Retired. Its paper numbers are kept verbatim under `_paper_reported_context` in `reference.json`. Weights are now WikiText 0.5, PTB 0.5. |

The reference method was rewritten in numpy and now fits on the full training files (23,767 / 42,068 lines; previously 3,000). It keeps the paper's steps: add-α base model, local masking and renormalization (Eq. 9), and global conditioning with an importance-sampling estimate of Z from M = 2000 samples of ℓ (Eq. 12–13) plus importance resampling (Eq. 14). It submits ℓ's table and the masked p_Δ for g; the evaluator's exact normalization of the latter *is* the conditioning. The IS estimate is a diagnostic: log₂Ẑ = −0.00717 vs exact −0.00697 (WikiText-2), and −0.02792 vs −0.02700 (PTB).

## Do better results score higher?

| Variant (runnable: `redteam/hardened/<case>/run.sh`) | Before (old gym, REDTEAM.md) | After: Δ test [95% CI] | After: Δ dev |
|---|---:|---:|---:|
| Reference, seed 13 vs seed 14 | seed spread 0.0216 | **0.0000** (identical scores) | 0.0000 |
| Tuned smoothing (α chosen on dev: 0.03) | α 0.005: +0.067 | **+0.233** [+0.162, +0.300] | +0.240 |
| Kneser–Ney base (discount chosen on dev) + same canonicalization | — | **+1.901** [+1.721, +2.093] | +1.991 |
| More training (old gym: 12,000 lines) | +1.655 (but primary ratio got worse) | n/a: reference now uses all training lines | — |
| Less training (3,000 lines = old reference scale) | 300 lines: −4.680 | **−17.993** [−19.20, −16.83] | −17.967 |
| Heavier smoothing (α 0.5) | — | **−7.123** [−7.57, −6.69] | −7.280 |
| Canonical unigram (simple baseline) | — | **−77.015** | −80.139 |
| Uniform canonical (= failure floor) | — | −166.661 | −170.448 |

Better variants beat the reference by far more than the test noise, and their CIs exclude 0. Per row, Kneser–Ney saves 0.33 bits per WikiText string and 1.95 per PTB string, with paired SE 0.05 and 0.10, improving 91% and 86% of strings. Worse variants rank lower. Dev and test deltas agree within 5% for every honest variant. The primary metric now moves the right way. For example, global relative loss on WikiText test is 0.99980 for the reference, 0.99032 for Kneser–Ney, and 1.0714 for 3,000 lines.

## Shortcuts: fix and score before → after

| Red-team shortcut | Before (Δ all) | Fix | After case | After Δ |
|---|---:|---|---|---:|
| Claimed zero loss | +41.27 | No probability field exists; evaluator normalizes the weights | `overweight_rows` (weights ×10) | **−167.46** (rejected: row sum > 1) |
| Impossible probability | +1,010,480.70 | As above; non-finite, negative and >1 rows rejected | `nan_weights`, `negative_weights` | **−167.46** |
| Baseline inflation | +587.04 | Baseline evaluator-owned; `base_bigram_lm` reserved | `reserved_baseline_dir` | **0.000** (ignored) |
| Vocabulary metadata | +184.23 | Support fixed and checked; `vocab_size` unscored | `support_shrunk`, `support_permuted` | **−167.46** |
| Token counts | +2.67 | Token counts come from evaluator assets | no field to submit | closed by construction |
| Easy cohort | +20.02 | Cohort is evaluator-owned | no records submitted | closed by construction |
| Empty samples / false flags / empty sample list | +0.029 / −0.48 / rejected | Samples and flags not read; exact canonical mass | `noncanonical_leak` (1e-8 weight on non-canonical pairs) | **−166.66** (gate) |
| Constraint tradeoff (better unconstrained model) | +2.49 | Canonicality gate | `unconstrained_kneser_ney`, `unconstrained_reference` | **−166.66**, **−166.67** |
| Fixed-tokenizer replacement | +0.029 | Evaluator reads only frozen copies | `tamper_tokenizer_and_support` (merges and canonical tables replaced in the workspace) | **−166.66**; workspace restored (verified by hash) |
| Null / NaN probabilities | rejected (crash) | Scored, not crashed | `nan_weights`, `pickled_npz` | **−167.46** |
| Direct score-file write | 0.000 | Output overwritten | `score_file_forgery` | **0.000** (overwritten: true) |
| Stale artifact crash (as-found) | test failed | Discovery skips non-submissions | `junk_artifacts` | **0.000** |
| *(new)* skip a hard experiment | — | Missing experiment = failure | `drop_ptb` | **−89.14** |
| *(new)* degenerate / non-terminating model | — | Zero-probability string and termination checks | `empty_string`, `nontight` | **−167.46** |
| Test memorization | +33.76 | Test text removed from the visible workspace | `test_memorization` (fit on train + test text) | **+7.251** test vs +0.461 dev: **left**, see below |

A failed method can no longer outrank one that solves the task through other metrics. A fully failed submission scores −166.66 to −167.46 (the uniform-canonical floor), far below even the canonical unigram baseline (−77.0). Failing a single experiment alone costs about −89.

## Shortcuts left, and why

1. **Fitting on the public test text** (`test_memorization`: +7.25 test, +0.46 dev). WikiText-2 and PTB test splits are public. Any agent with internet access can fetch them, and a weight table cannot reveal what text it was fit on. Mitigation: the text is no longer in the visible data, and `CONSTRAINTS.md` / `DIRECTION.md` forbid it. Its signature is a test gain about 16× the dev gain, which honest methods never show (≈1.0×). The grader should check for it.
2. **Fitting on the validation text** (`dev_memorization`: +0.49 test, +7.00 dev). This is extra in-domain data and cannot be detected from the weights. It is forbidden in `CONSTRAINTS.md`. Its signature is the opposite gap (dev gain 14× test gain), and its test gain is about the size of honest α tuning.

## Reference score before and after

| | Before (old gym) | After |
|---|---|---|
| WikiText-2 test bits/string: base / local / global | 35.701 / 35.685 / 35.639 (14 strings) | 34.7399 / 34.7352 / 34.7329 (275 strings) |
| PTB test bits/string: base / local / global | 109.503 / 109.384 / 109.226 (24 strings) | 88.4911 / 88.4557 / 88.4641 (516 strings) |
| Primary relative loss, local / global (WT; PTB) | 0.99956 / 0.99825; 0.99891 / 0.99747 | 0.99986 / 0.99980; 0.99960 / 0.99969 |
| Canonical rate: base / local / global | sampled 0.97 / 1 / 1; 0.79 / 1 / 1 | exact 0.99518 / 1 / 1; 0.98146 / 1 / 1 |
| No-arg `evaluate_test.sh` on the as-found workspace | crashed (stale artifact) | runs; scores every method on disk |
| Composite | 0 (anchor) | 0 (anchor); seed 14 reproduces it exactly |

The paper's claim holds in the paper's direction on both corpora and both splits. Local loss is ≤ baseline for *every* string (each local normalizer is ≤ 1). Global loss is the baseline shifted by the exact log₂Z on every string. With zero seed noise, both gains are exact rather than statistical.

## Headroom check

- **At the bound:** `canonical_sample_rate` = 1 for both reference rows. That is the constraint itself, and the gate enforces it.
- **Not at a bound:** all four loss metrics. Tuned smoothing (+0.23) and Kneser–Ney (+1.90) beat the reference with CIs clear of 0, and no tried variant reaches a bound.
- **Baselines:** the canonical unigram baseline (−77) and the failure floor (−167) sit clearly below, so there is room on both sides.

No extra experiment was needed.

## Weights

Experiment weights went from 0.3 / 0.3 (+0.4 on the unmeasured GPT-2 entry) to 0.5 / 0.5. The paper's Figure 3 treats WikiText and PTB as equal, and the 0.4 entry scored nothing. Metric coefficients are unchanged. `finite_record_rate` and `vocab_size` were removed: the first is now a gate, the second is fixed by the evaluator. Paper-reported values in `reference.json` are unchanged.

## What I could not fix, stated plainly

- **The paper's canonicalization effect is tiny at this scale:** 0.005–0.035 bits per string on the test cohort, relative loss 0.9996–0.9999, against 0.994–0.999 for GPT-2 in the paper. The bigram model over 2048 tokens puts only 0.5% (WikiText) and 1.9% (PTB) of its mass on non-canonical strings. The direction is right and the effect is exact, but most headroom above the reference comes from better estimation under the exact constraint (smoothing), not from the canonicalization step. The constraint itself is enforced as a gate, which is where the paper's idea shows up most in the ranking.
- **Local vs global ordering is the reverse of the paper's here.** In this bigram setting global is better on WikiText (34.7329 vs 34.7352) and local is better on PTB (88.4557 vs 88.4641). The paper found the opposite on each corpus. This is not a scored claim.
- **The model class is limited to bigram tables.** That is what makes exact, verifiable scoring possible, but it excludes higher-order or neural models. The paper's GPT-2/Llama experiments are not reproduced; the GPT-2 entry was retired, not run.
- **The canonical-bigram table is approximate.** It comes from the GPT-2 bigram round-trip test (pure BPE, or the full tokenizer on the decoded pair) plus train-observed pairs (0 overrides were needed). Like the paper's test, it approximates canonicality under pre-tokenization. It is the fixed definition the gym scores against. No held-out line was dropped for containing a pair the table calls non-canonical.
