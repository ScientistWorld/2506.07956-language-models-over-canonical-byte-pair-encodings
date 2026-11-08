# Progress

## Current state (iteration 11 hardening; see `HARDENING.md`)

- Task: on WikiText-2 and PTB, build a token bigram LM over a fixed 2048-token GPT-2 BPE support (`data/support/`) that puts all its probability on canonical token strings. Submissions are weight tables (`artifacts/<method>/<experiment>.npz`). The evaluators normalize them exactly and score held-out lines fully inside the support.
- Reference (`scripts/run.sh` → `method/canonical_conditioning.py`): add-0.05 bigram on the full training file; local canonicalization (masked, renormalized); global canonicalization (masked p_Δ, normalized exactly by the evaluator); importance-sampling estimate of log Z with 2000 local samples as a diagnostic (WikiText-2 −0.00717 vs exact −0.00697; PTB −0.02792 vs −0.02700).
- Hidden test (`scoring/scores_test.json`), bits/string base / local / global: WikiText-2 34.7399 / 34.7352 / 34.7329; PTB 88.4911 / 88.4557 / 88.4641. Canonical mass: base 0.9952 / 0.9815, both canonical models 1.0.
- Development (`scoring/scores_train.json` = `scores.json`): WikiText-2 33.1355 / 33.1305 / 33.1285; PTB 90.8112 / 90.7759 / 90.7842.
- Limitation: the canonicalization gain is small at bigram scale (0.005–0.035 bits/string), so headroom comes mostly from better estimation under the exact constraint. The GPT-2 experiment was retired unrun.

## History (earlier iterations; superseded numbers)

## What Works

- GPT-2 BPE tokenizer files and WikiText-2 data are downloaded into `data/downloads/`.
- The method implements the paper's conditioning family: base token LM, local canonicality masking, local importance weights, and global log-loss adjustment from an estimated canonicality rate.
- `scripts/method.sh` writes a reusable artifact and `scripts/evaluate.sh` aggregates it without importing `method/`.

## Results

- Tiny smoke test completed on 50 train lines and 10 test lines. It was used only to validate the pipeline.
- The completed scaled WikiText-2 benchmark trained on 1000 lines, evaluated 50 held-out lines, used an expanded 1302-token GPT-2-BPE support, and sampled 100 baseline/local strings for canonicality/global normalization checks.
- Superseded pre-audit scores from the earlier run were:
  - `base_bigram_lm`: 649.346 bits/string, measured canonical sample rate 0.88, vocab size 1302.
  - `global_canonical_bigram_lm`: 649.129 bits/string, canonical sample rate 1.0, vocab size 1302.
  - `local_canonical_bigram_lm`: 649.039 bits/string, canonical sample rate 1.0, vocab size 1302.
- The result supports the core claim in the scaled setting: canonical conditioning removes invalid token strings and improves held-out log-loss without changing vocabulary size.
- `python validate.py --compare` passes all structure and score checks.
- 2026-06-13 continuation audit confirmed the paper-specific computation is present in `method/canonical_conditioning.py`, but a parallel audit found protocol issues in the previous run. The milestone was held at `method_runs` until the corrected benchmark job regenerated `scoring/scores.json`.
- Corrected CPU benchmark `b53a337b-9ca` completed successfully after those fixes. It used 3000 WikiText-2 train lines, scanned 200 test lines, scored the 20 held-out lines fully representable under the 2048-token train-derived support, and drew 200 completed baseline/local samples.
- Corrected scores in `scoring/scores.json`:
  - `base_bigram_lm`: 36.902 bits/string, measured canonical sample rate 0.97, vocab size 2048.
  - `global_canonical_bigram_lm`: 36.858 bits/string, canonical sample rate 1.0, vocab size 2048.
  - `local_canonical_bigram_lm`: 36.885 bits/string, canonical sample rate 1.0, vocab size 2048.
- The corrected run restores `core_claim`: both canonicalized variants improve held-out log-loss while satisfying the canonical sample constraint.
- Larger CPU follow-up `9adc68d5-926` completed successfully. It used 6000 WikiText-2 train lines, scanned 600 held-out lines, scored the 92 held-out lines fully representable under the 4096-token train-derived support, and drew 300 completed baseline/local samples.
- Current `scoring/scores.json` from that follow-up:
  - `base_bigram_lm`: relative log-loss 1.0, 40.380 bits/string, measured canonical sample rate 0.90, vocab size 4096.
  - `global_canonical_bigram_lm`: relative log-loss 0.996584, 40.242 bits/string, canonical sample rate 1.0, vocab size 4096.
  - `local_canonical_bigram_lm`: relative log-loss 0.999570, 40.362 bits/string, canonical sample rate 1.0, vocab size 4096.
- The follow-up reaches `core_claim_plus`: the corrected canonical conditioning result holds at a broader train-derived support and on more held-out lines.
- Final verification retry `cf6dce5d-31d` was killed by Hive/Ray heartbeat disk-quota errors and orphaning before aggregate scores were regenerated. The WikiText artifact from the completed follow-up was inspected and already contains generated samples, and it scores correctly under the hardened evaluator.
- The next retry uses the same algorithm with cached allowed-candidate lists and a smaller two-corpus CPU configuration: 3000 train lines, 2048-token train-derived support, and 100 completed samples per corpus.
- CPU test retry `8243ca38-61e` completed successfully. It regenerated both WikiText-2 and PTB artifacts, refreshed `scoring/scores.json`, and passed `python validate.py --compare`.
- Final regenerated scores:
  - `wikitext2_bigram_conditioning/base_bigram_lm`: relative log-loss 1.0, 36.688 bits/string, canonical sample rate 0.97, vocab size 2048.
  - `wikitext2_bigram_conditioning/local_canonical_bigram_lm`: relative log-loss 0.999570, 36.672 bits/string, canonical sample rate 1.0, vocab size 2048.
  - `wikitext2_bigram_conditioning/global_canonical_bigram_lm`: relative log-loss 0.998296, 36.626 bits/string, canonical sample rate 1.0, vocab size 2048.
  - `ptb_bigram_conditioning/base_bigram_lm`: relative log-loss 1.0, 102.495 bits/string, canonical sample rate 0.79, vocab size 2048.
  - `ptb_bigram_conditioning/local_canonical_bigram_lm`: relative log-loss 0.998857, 102.377 bits/string, canonical sample rate 1.0, vocab size 2048.
  - `ptb_bigram_conditioning/global_canonical_bigram_lm`: relative log-loss 0.997294, 102.217 bits/string, canonical sample rate 1.0, vocab size 2048.
- 2026-06-13 final verification kept the scored gym baseline on the verified two-corpus scaled run. A cached GPT-2-small runner remains available in `method/gpt2_conditioning.py` for future extension, but it has not produced verified scores and is not part of the current `scores.json`.

## Remaining

- A future extension can run the optional GPT-2-small implementation on compute and add a verified `wikitext2_gpt2_small_conditioning` score if it improves held-out likelihood while reaching tokenizer-valid generated support.

## Issues

- The paper's linked GitHub repository currently returns 404, so this reproduction implements the algorithm directly from the paper.
- The original paper uses GPT-2/Llama neural LMs; this first runnable environment scales the LM down to a token bigram model to keep the first job CPU-safe.
- First submitted CPU job was cancelled after 20 minutes because it produced no method progress output; the rerun uses a smaller benchmark slice and more frequent progress logging.
- The first completed job showed global canonicalization improving log-loss, but local canonicalization assigned `-inf` to 17/50 strings due to GPT-2 pre-tokenizer false negatives. The next run adds the paper's corpus-derived override workaround for observed canonical bigrams.
- The final completed CPU run fixed the false negatives using those overrides and produced finite improved losses for both proposed conditioning variants.
- A later audit found the baseline `canonical_sample_rate` was hardcoded. This was fixed by sampling the unconstrained bigram LM and aggregating the measured sample-level canonicality from the method artifact.
- The 2026-06-13 audit found that `eval/aggregate_scores.py` silently skipped non-finite log probabilities, which could hide a broken run. The evaluator now fails on missing or non-finite per-string log-probability fields; the current artifact still validates under the stricter check.
- The same audit found that the previous run used test-set bigrams as canonical overrides and included test tokens in the reduced vocabulary. The corrected method uses a train-derived vocabulary, scores only held-out lines fully representable in that fixed support, uses tokenizer-reference canonical bigram checks without reading the test set, and restricts corpus-derived overrides to training sequences.
- The baseline script previously shelled into `method/`; it is now an independent unconstrained bigram implementation using only `data/` helpers.
- Global canonical sample rate was hardcoded in `eval/`; the corrected method writes an artifact-level measured rate from importance-resampled completed local samples, and `eval/` reads that field.
- Current highest honest milestone is `core_claim_plus`; remaining gaps are secondary paper settings rather than the core conditioning claim.
- The primary scoring metric was changed from absolute `log_loss_bits_per_string` to `relative_log_loss_to_baseline` after validation flagged that the paper's GPT-2-small WikiText strings and this scaled WikiText-2 bigram benchmark have different absolute bits/string scales. Absolute bits/string remains reported as a secondary metric.
- The evaluator now verifies `canonical_sample_rate` from generated token samples stored in the artifact instead of trusting method-reported metadata. This keeps the constraint metric benchmark-side while preserving the `eval/` versus `method/` separation.
- PTB train/valid/test URLs from `wojzaremba/lstm` were verified with HTTP 200 responses and added to `scripts/download.sh`; the files are downloaded into `data/downloads/` for compute nodes.
- CPU job `cf6dce5d-31d` failed with exit code `-9` after repeated Hive/Ray heartbeat `Disk quota exceeded` errors. The retry reduces runtime and log volume but does not treat the infrastructure failure as a reason to lower the milestone.
- The optional train/test evaluators initially merged methods globally and conflicted when both corpora reused the same method names. They now aggregate by experiment, and refreshed `scores_train.json` and `scores_test.json` both cover WikiText-2 and PTB.
- A follow-up audit flagged that the successful reproduction was still scaled to a smoothed bigram LM and covered a small filtered subset. An optional GPT-2-small implementation was added, but final verification did not include it in default scoring because no completed job had verified its scores.
- The generated `jobs/` directory had been tracked despite `.gitignore`; it has been removed from the index and `jobs/` is now ignored.
- Final verification found that the scaled bigram `reference.json` entries reused paper GPT-2 absolute losses, making reproduced absolute losses appear roughly 10x off. The scaled experiments now keep only paper-derived relative-loss context in `reference.json`; scaled absolute losses and constraint metrics are produced by `scores.json`.

## Deviations from Paper

- Paper used GPT-2 and Llama neural language models; this first reproduction uses a smoothed token bigram LM over GPT-2 BPE tokens. The canonicality algorithm itself is unchanged: it still masks noncanonical BPE extensions, computes local weights, and estimates global conditioning from local samples.
- Paper evaluated full PTB/WikiText test splits; this scaled CPU reproduction uses WikiText-2 and PTB subsets to keep the environment short and debuggable.
- The GPT-2-small follow-up still uses a small WikiText-2 subset and a short length limit for canonicality-rate sampling so it can complete inside the remaining budget. The conditioning computation is unchanged.
- The implementation handles GPT-2 pre-tokenization by first checking the paper's pure BPE bigram round trip and then checking the full GPT-2 tokenizer on valid UTF-8 token pairs. Corpus-derived overrides are allowed only from training sequences, not held-out text.
- The reduced vocabulary is derived from training text only. Held-out lines containing out-of-support tokens are omitted from the scaled score rather than adding test tokens to the model support.
