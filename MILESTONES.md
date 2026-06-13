# Reproduction Milestones

**Current: core_claim**

## Progress Log

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
