# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

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
