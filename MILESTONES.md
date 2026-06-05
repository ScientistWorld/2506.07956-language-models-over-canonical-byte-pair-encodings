# Reproduction Milestones

**Current: core_claim**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-06-05 08:05] - none
- Read the paper and identified the core canonical BPE conditioning claim.
- Confirmed the linked code repository is currently unavailable.

### [2026-06-05 08:12] - method_runs
- Implemented GPT-2 byte-level BPE tokenization, BPE bigram canonicality testing, local canonicalization, and global canonicality-rate estimation.
- Ran a tiny end-to-end smoke test that produced method artifacts and scores.

### [2026-06-05 08:41] - core_claim
- Completed the scaled WikiText-2 CPU reproduction with corpus-derived GPT-2 pretokenizer canonical bigram overrides.
- The unconstrained bigram LM scored 649.346 bits/string; global canonical conditioning improved to 649.183 bits/string and local canonical conditioning improved to 649.039 bits/string.
- Both canonical methods achieved `canonical_sample_rate=1.0`, satisfying the tokenizer-validity constraint while improving held-out log-loss.

## Stop Justification
- Completed at milestone `core_claim`.
