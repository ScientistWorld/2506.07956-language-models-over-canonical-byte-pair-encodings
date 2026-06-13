# Verification Report

## Initial Assessment

The workspace already contained a real scaled implementation of the paper's
canonical-conditioning computation and had reached `core_claim_plus` on a
WikiText-2 setting. The claimed milestone was justified, but final verification
found stale scoring artifacts and documentation/evaluation issues that needed
cleanup before the workspace could serve reliably as a gym.

## Issues Found

- **Critical:** the hardened evaluator required generated samples for
  benchmark-side canonicality verification, while the final aggregate scores
  had not yet been regenerated with both WikiText-2 and PTB artifacts.
- **Critical:** optional train/test evaluators merged method names globally,
  which broke when multiple experiments reused the same method identifiers.
- **Minor:** method-agnostic documentation leaked implementation details in
  problem/evaluation/scoring guidance.
- **Minor:** the first verification retry was killed by Hive/Ray heartbeat
  disk-quota/orphaning failures before PTB completed.

## Fixes Applied

- Cleaned method-agnostic docs in `briefing/` and `scoring/`.
- Hardened `eval/aggregate_scores.py` and train/test evaluators to verify
  canonical sample rates from generated token samples instead of trusting
  method metadata.
- Cached allowed next-token candidates in `method/canonical_conditioning.py`
  and scaled `scripts/run.sh` to a fast CPU two-corpus configuration.
- Regenerated `scoring/scores.json`, `scoring/scores_train.json`, and
  `scoring/scores_test.json` from completed job `8243ca38-61e`.
- Fixed train/test aggregation so results are separated by experiment.

## Final Judgment

The workspace is ready for gym use. Improve-mode agents can run the packaged
method and compare against refreshed scores. From-scratch agents can use the
problem, data, and evaluation contract without importing `method/`. The key
limitation is scale: the environment uses a CPU-safe smoothed token bigram LM
and corpus subsets rather than the full neural-LM experiments from the paper.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: high
- Key limitation: scaled bigram LM and subset evaluation rather than full neural-LM scale
