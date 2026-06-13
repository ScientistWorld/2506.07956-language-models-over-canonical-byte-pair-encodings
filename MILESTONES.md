# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-05 04:44] - core_claim
- Completed CPU reproduction of canonicality by conditioning on a scaled WikiText-2 token bigram language model.
- The method artifact includes the paper's local canonicalized model, global canonicalized log-loss from an importance-sampled canonicality rate, and an unconstrained baseline.
- `scoring/scores.json` shows both canonicalized variants improve held-out log-loss and reach canonical sample rate 1.0 while the unconstrained baseline has lower canonical sample rate.
