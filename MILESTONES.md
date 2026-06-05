# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-05 08:47] - core_claim
- Audited the existing implementation and confirmed it implements the paper's canonicality-by-conditioning computation: token LM probabilities, BPE canonical bigram masking, local renormalization, local importance weights, and global conditioning estimate.
- Corrected the constraint metric so the unconstrained baseline canonical sample rate is measured from generated samples rather than hardcoded.
- Completed a CPU scheduler run on the scaled WikiText-2 benchmark with 1000 train lines, 50 held-out lines, 1302 GPT-2-BPE tokens, and 100 samples.
- Reproduced the core claim in the scaled setting: local and global canonicalization improve held-out log-loss while producing only canonical samples.

## Stop Justification
- Completed at milestone `core_claim`.
