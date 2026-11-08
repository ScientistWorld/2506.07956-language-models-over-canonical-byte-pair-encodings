# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-11-08] - core_claim_plus (iteration 11 hardening)
- Rebuilt the test so it scores a submitted model, not method-written numbers. Each submission is a weighted bigram table over a fixed 2048-token GPT-2 support. The evaluator normalizes it exactly, scores hidden test strings (275 WikiText-2 and 516 PTB, up from 14 and 24), and computes the exact canonical mass, gating anything above 1e-9.
- Reference rerun on the full training files: local and global canonicalization lower held-out bits/string on both corpora and both splits (WikiText-2 test 34.7399 → 34.7352 / 34.7329; PTB 88.4911 → 88.4557 / 88.4641) with canonical mass exactly 1. Scores are deterministic: seed 14 reproduces them exactly.
- Variants rank correctly (tuned smoothing +0.23, Kneser–Ney +1.90, 3,000 lines −17.99). All red-team shortcuts fail or tie, except fitting on externally obtained test text (see `HARDENING.md`).

### [2026-06-13 09:25] - core_claim_plus
- Final verification confirmed the current completed gym baseline is the verified two-corpus scaled run, not the optional unverified GPT-2-small follow-up.
- Cleaned method-agnostic target docs and reference entries so `scores.json` experiments are the active gym surface and scaled absolute losses are not compared against unrelated paper GPT-2 absolute losses.
- Kept milestone at `core_claim_plus`: the method implements the paper's canonical-conditioning computation and demonstrates the core likelihood/validity claim on WikiText-2 and PTB scaled settings.

### [2026-06-13 09:07] - core_claim_plus
- Follow-up audit identified the main remaining fidelity gap: the current successful result uses a scaled token bigram LM rather than the paper's GPT-2/Llama neural LMs.
- Added a cached GPT-2-small canonical-conditioning implementation for future verification; no higher milestone is claimed because no completed job has produced inspected GPT-2-small scores.
- Removed generated `jobs/` logs from git tracking and fixed wrapper scripts so custom output paths do not accidentally evaluate stale artifacts.

### [2026-06-13 08:58] - core_claim_plus
- CPU test retry `8243ca38-61e` completed successfully and regenerated both WikiText-2 and PTB artifacts using the scaled canonical-conditioning implementation.
- `scoring/scores.json` now contains both `wikitext2_bigram_conditioning` and `ptb_bigram_conditioning`; in both corpora, local and global conditioned variants improve held-out log-loss over the unconstrained baseline and reach canonical sample rate 1.0.
- Standalone evaluation, train/test slice evaluation, and `python validate.py --compare` all pass after fixing train/test aggregation to separate methods by experiment.

### [2026-06-13 08:51] - core_claim_plus
- CPU verification job `cf6dce5d-31d` was killed after Hive/Ray heartbeat disk-quota failures and orphaning, before it could finish PTB and regenerate aggregate scores.
- The existing WikiText artifact was checked and already contains generated samples required by the hardened evaluator; it scores successfully under `eval/aggregate_scores.py`.
- Added allowed-candidate caching to the local conditioned sampler and scaled `scripts/run.sh` to a faster two-corpus CPU verification setting for retry.

### [2026-06-13 08:21] - core_claim_plus
- Prepared a secondary corpus reproduction: PTB download paths verified and added, method/baseline artifacts made experiment-name aware, and the aggregate evaluator now supports multiple experiment artifacts.
- Fixed a benchmark integrity issue by requiring generated token samples in artifacts and independently verifying canonical sample rate inside `eval/` instead of trusting method metadata.
- Submitted a CPU job to rerun WikiText-2 and PTB canonicality-by-conditioning experiments before claiming `secondary_claims`.

### [2026-06-13 07:53] - core_claim_plus
- Completed the larger CPU follow-up `9adc68d5-926` using 6000 WikiText-2 train lines, 600 scanned held-out lines, a 4096-token train-derived support, and 300 completed local samples.
- The run scored 92 held-out lines representable under the fixed support and passed `python validate.py --compare`.
- Results strengthen the core claim at the broader setting: baseline 40.380 bits/string with canonical sample rate 0.90; local canonicalization 40.362 bits/string with rate 1.0; global canonicalization 40.242 bits/string with rate 1.0.

### [2026-06-13 07:37] - core_claim
- Completed corrected CPU benchmark after audit fixes: train-only reduced vocabulary, train-only canonical bigram overrides, completed local samples for global normalizer estimation, independent baseline, and artifact-measured canonicality rates.
- Corrected run used 3000 WikiText-2 train lines, scanned 200 held-out lines, scored 20 held-out lines representable under the 2048-token train-derived support, and used 200 completed local samples.
- Results restored the core claim: baseline 36.902 bits/string with canonical sample rate 0.97; local canonicalization 36.885 bits/string with rate 1.0; global canonicalization 36.858 bits/string with rate 1.0.

### [2026-06-13 07:23] - method_runs
- Audited the continued workspace and confirmed the method implements canonicality by conditioning rather than a surrogate.
- Parallel audit found protocol issues in the previous `core_claim` run: test-derived canonical bigram overrides, test-derived vocabulary support, truncated samples in the global normalizer, a method-dependent baseline script, and a hardcoded global canonicality score.
- Fixed those issues and smoke-tested the corrected method and independent baseline on a tiny local slice; the larger corrected CPU benchmark is being submitted before restoring `core_claim`.

### [2026-06-05 04:44] - core_claim
- Completed CPU reproduction of canonicality by conditioning on a scaled WikiText-2 token bigram language model.
- The method artifact includes the paper's local canonicalized model, global canonicalized log-loss from an importance-sampled canonicality rate, and an unconstrained baseline.
- `scoring/scores.json` shows both canonicalized variants improve held-out log-loss and reach canonical sample rate 1.0 while the unconstrained baseline has lower canonical sample rate.
