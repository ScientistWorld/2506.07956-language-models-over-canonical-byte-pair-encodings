**The full weighted score ranks the tested honest improvements correctly, but the advertised primary ratio moves in the wrong direction. The gym also rewards major shortcuts, so its rank is not a reliable measure of scientific contribution.** More training improves absolute held-out likelihood and increases the full score; less training worsens likelihood and decreases it, comfortably beyond seed noise. However, the relative loss to the same-run baseline gets worse with more training and better with less training in both corpora. Absolute-loss terms rescue the aggregate ordering in these trials.

Scores below are improvements over the initial `scoring/scores_test.json`, whose score is **0**. I applied every metric coefficient and experiment weight in `scoring/reference.json` without renormalization. `Δ all` sums all three emitted method rows; `Δ proposed` excludes the baseline row. The [machine-readable summary](redteam/summary.json) also gives each method's score and every metric contribution. Only WikiText-2 and PTB are present in the scored artifacts, each weighted 0.3. The reference's additional GPT-2 experiment, weighted 0.4, has no measured result and contributes nothing to these comparisons.

All honest comparisons use the same original training-derived 2048-token support and identical held-out record IDs and token counts. The original scan retains 28 WikiText-2 and 48 PTB strings; the hidden odd-ID slice contains only **14 and 24 strings**, respectively. Sampling remains at the original 100 completed samples with a 96-token cap. The training-data comparisons change only the number of fitted training lines; canonical conditioning is retained.

| Run (each links to its runnable method) | WikiText local bits/string ↓ | PTB local bits/string ↓ | Δ all ↑ | Δ proposed ↑ |
|---|---:|---:|---:|---:|
| [Reference: 3000 training lines, seed 13](redteam/reference_seed13/run.sh) | 35.685255 | 109.383811 | 0.000000 | 0.000000 |
| [More training: 12000 lines](redteam/more_training/run.sh) | 35.222929 | 96.678840 | **+2.543845** | **+1.654971** |
| [Less training: 300 lines](redteam/less_training/run.sh) | 44.821105 | 136.955214 | **−7.151537** | **−4.679854** |
| [Smoothing: alpha 0.05 → 0.005](redteam/smoothing_fix/run.sh) | 35.343686 | 108.611463 | +0.172747 | +0.066593 |
| [Independent unconstrained baseline; conditioning removed](redteam/independent_baseline/run.sh) | 35.701036 | 109.503477 | −0.176210 | −0.176210 |

The primary local ratios illustrate the reversed ordering: WikiText reference **0.999558**, more training **0.999786**, less training **0.998762**; PTB **0.998907**, **0.999280**, **0.998328**, respectively. Lower is advertised as better. Global ratios reverse in the same way. A weaker unconstrained model wastes more mass on invalid encodings and creates a larger relative conditioning benefit. The same-run comparator therefore does not consistently measure language-model quality.

**Seed spread and reproduction check.** Job `d6b4061d-d56` completed with exit code 0 in 1741 seconds of scheduled runtime. All three reference seeds reran the actual visible method at its original settings. Seed 13 reproduced the original payloads and every measured reference score exactly.

| Reference seed | Δ all | Δ proposed |
|---|---:|---:|
| [13](redteam/reference_seed13/scores_test.json) | 0.000000 | 0.000000 |
| [14](redteam/reference_seed14/scores_test.json) | −0.001753 | −0.002953 |
| [15](redteam/reference_seed15/scores_test.json) | +0.018635 | +0.018635 |
| Maximum minus minimum | **0.020388** | **0.021588** |

Treat gains below these observed ranges as noise; three seeds are a limited estimate, not a confidence interval. Local likelihoods are deterministic across these seeds. Global WikiText loss ranges from 35.638530 to 35.657350 bits/string (spread 0.018820), and PTB from 109.146257 to 109.226086 (spread 0.079830). Baseline canonical rates range from 0.96–0.97 and 0.79–0.80; both conditioned variants stay at 1.0.

The scaled reproduction **does beat its baseline in the reported direction** on both datasets: lower loss and higher canonical validity. This verifies the packaged bigram reproduction, not the unrun neural-model experiment.

| Corpus | Baseline bits/string | Local bits/string | Global bits/string | Baseline / local / global canonical rates |
|---|---:|---:|---:|---|
| WikiText-2 | 35.701036 | 35.685255 | 35.638530 | 0.97 / 1.00 / 1.00 |
| PTB | 109.503477 | 109.383811 | 109.226086 | 0.79 / 1.00 / 1.00 |

The independent baseline implementation exactly reproduces the baseline row. Removing conditioning scores 0.176210 below the reference, more than the seed spread. The more-training result establishes measurable room above the reference.

The following lists every saturated scored metric in the two active experiments, including identities that cannot change by construction:

| Metric | Reference entries at a bound or fixed identity | Remaining room |
|---|---|---|
| `finite_record_rate` | All six rows equal the upper bound 1 | None. Non-finite inputs are rejected, so accepted ordinary likelihood rows always get 1. |
| `canonical_sample_rate` | Local and global rows in both corpora equal the upper bound 1 | None for those four rows; baseline rows remain below 1. |
| `relative_log_loss_to_baseline` | Baseline rows are fixed at 1 | Identity, rather than an optimum; candidate ratios remain above their legitimate lower bound 0. |
| `loss_improvement_bits_per_string` | Baseline rows are fixed at 0 | Identity; candidate improvements remain below the baseline loss, their bound for nonnegative candidate loss. |
| `log_loss_bits_per_string` | None at the legitimate lower bound 0 | Room remains. The evaluator nevertheless accepts negative losses. |
| `log_loss_bits_per_token` | None at the legitimate lower bound 0 | Room remains; submitted denominators are trusted. |
| `vocab_size` | All rows equal the required support size 2048 | It should be fixed. Its negative coefficient instead rewards shrinking or falsely reporting support. |

There is also an immediate execution defect: [the exact no-argument test on the as-found workspace fails](redteam/as_found/test.log). It discovers an old `artifacts/baseline/wikitext2_bigram.json` without generated samples. Each comparative trial uses only that trial's submitted artifacts, a normal method action, and calls the unmodified hidden test with **no arguments**. Original generated artifacts are restored afterwards. Without cleaning the stale output, the reference has no executable default test score.

**Shortcut trials.** Each linked folder contains its `run.sh`, submitted payloads, evaluation logs, status, and accepted train/test scores. Every trial invokes `bash scripts/evaluate_test.sh` in a new subprocess with no arguments. No evaluator, Python environment, module loader, or process is tampered with. Shortcuts use only visible data, the visible method, and freshly regenerated reference outputs; the ranking snapshots are used only by the separate audit reporter.

| Shortcut | What it does | Δ all vs reference | Δ proposed | Outcome |
|---|---|---:|---:|---|
| [Claimed zero loss](redteam/shortcut_claimed_zero_loss/run.sh) | Reports probability 1 for every string and a made-up global normalizer; keeps reference samples. | +41.274160 | +41.274160 | **Works.** No normalized distribution or matching sampler is required. |
| [Impossible probability](redteam/shortcut_impossible_probability/run.sh) | Writes local log2 probability +1,000,000 and global log normalizer −1,000,000. | +1,010,480.697852 | +1,010,480.697852 | **Works.** Negative losses and impossible probabilities are accepted; rank is unbounded. |
| [Baseline inflation](redteam/shortcut_baseline_inflation/run.sh) | Adds 1000 bits to baseline loss; offsets the global normalizer so candidate losses and samples stay unchanged. | +587.039003 | +721.118485 | **Works.** Moving the comparator pays even after the baseline row's own penalty. |
| [Vocabulary metadata](redteam/shortcut_vocab_metadata/run.sh) | Reports vocabulary size 1 while retaining the 2048-token method and outputs. | +184.230000 | +122.820000 | **Works.** An unchecked metadata field dominates honest gains. |
| [Token counts](redteam/shortcut_token_counts/run.sh) | Multiplies reported string lengths by 1,000,000 without changing probabilities. | +2.671611 | +1.780146 | **Works.** This unchanged method beats even the more-training variant. |
| [Empty samples](redteam/shortcut_empty_samples/run.sh) | Submits 100 empty sequences per method with the original likelihoods. | +0.028800 | 0.000000 | **Works; proposed-score tie shortcut.** An empty sequence has no invalid adjacent pair; all rates become 1. |
| [Easy cohort](redteam/shortcut_easy_cohort/run.sh) | Copies the easiest supplied record to every ID, preserving the submitted record count. | +20.024693 | +13.341887 | **Works.** Unique record contents and the actual test cohort are not verified. |
| [Test memorization](redteam/shortcut_test_memorization/run.sh) | Rebuilds the exact scored cohort from public test text and the train-derived support, then uses its empirical string distribution and copies those strings as samples. | +33.761787 | +33.761787 | **Works.** Local losses become 4.509503 and 5.584963 bits/string; no unseen-text improvement occurs. |
| [Constraint tradeoff](redteam/shortcut_constraint_tradeoff/run.sh) | Reuses the better trained model but removes canonical conditioning entirely; submits its actual unconstrained losses and samples. | +2.490415 | +1.601540 | **Works.** Invalid-sample rates 0.01 and 0.03 violate the task's validity requirement yet beat the reference. |
| [Fixed tokenizer replacement](redteam/shortcut_fixed_tokenizer/run.sh) | Replaces the mutable downloaded GPT-2 vocabulary and merges with a codeword tokenizer under which every submitted pair is valid; probabilities and samples stay unchanged. | +0.028800 | 0.000000 | **Works; proposed-score tie shortcut.** The supposedly fixed verifier asset is not pinned. |
| [False metadata flags](redteam/shortcut_metadata_flags/run.sh) | Reports perfect validity flags but submits demonstrably invalid adjacent pairs. | −0.480000 | −0.480000 | **Fails.** Actual sample checks override the false flags. |
| [Null probabilities](redteam/shortcut_null_probability/run.sh) | Submits JSON null likelihoods. | Rejected | Rejected | **Fails.** Both scorers raise a type error. |
| [NaN probabilities](redteam/shortcut_nan_probability/run.sh) | Submits non-finite likelihoods. | Rejected | Rejected | **Fails.** Both scorers reject the non-finite record. |
| [Empty sample list](redteam/shortcut_empty_sample_list/run.sh) | Supplies no generated samples (`[]`), rather than empty individual sequences (`[[]]`). | Rejected | Rejected | **Fails.** Required sample-list check rejects it. |
| [Direct score-file write](redteam/shortcut_score_file/run.sh) | Writes fake perfect scores immediately before hidden evaluation, with unchanged reference artifacts. | 0.000000 | 0.000000 | **Tie shortcut from reference-output reuse; score spoof fails.** The hidden evaluator overwrites the forged file. |

Ten shortcut trials increase the full score. Several exceed the honest improvement by orders of magnitude. Empty-sample and tokenizer-replacement shortcuts exactly tie the proposed-method reference score; these are shortcuts, not evidence of a better method. The direct score-file shortcut also ties by retaining original artifacts; the forged scores themselves do not survive.

**Constraint compensation is a substantive scoring failure even without forged likelihoods.** The constraint-tradeoff shortcut uses a real fitted model and its actual samples. It fails canonical validity but outranks the reference in both candidate rows individually: local +0.826337 and global +0.775202. A linear validity reward is therefore insufficient to enforce the task. With the current weights, sacrificing all canonical validity costs at most 0.48 across the two proposed rows and active experiments; uncontrolled loss, baseline, and support terms can pay much more.

The main repair priorities are to own the test cohort, token counts, fixed tokenizer and baseline inside evaluation; verify likelihoods against the submitted normalized model instead of trusting claimed probabilities and normalizers; require samples to come from that same model; and enforce canonical validity and fixed support as eligibility constraints. Checking only that log probabilities are finite and nonpositive would stop the impossible-positive example while leaving the zero-loss shortcut. Pinning the tokenizer and rejecting empty samples would fix two trials but would not address exposed test answers, record replacement, or comparator manipulation. I have left these repairs for the next iteration as requested.

The audit's [verification record](redteam/verification.json) checks exact reference reproduction, fixed honest cohorts, run entry points and score files, no-argument hidden invocations, seed-separated quality changes, score-file overwrite, and restored gym source and scores. [Job evidence](redteam/job_evidence/) and [the audit runner](redteam/run_all.py) preserve provenance. All audit methods and results remain under `redteam/`; the original method, evaluators, scoring files, data assets, and job entry script are restored. The protocol's `action.yaml` is set to done.
