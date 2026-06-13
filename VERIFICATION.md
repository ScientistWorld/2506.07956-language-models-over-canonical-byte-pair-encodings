# Verification Report

## Initial Assessment

The workspace started at `core_claim_plus` with a verified two-corpus scaled
reproduction in `scoring/scores.json`. The claimed milestone is justified: the
implemented method contains the paper's canonical-conditioning computation and
the completed WikiText-2/PTB scaled runs show lower held-out log-loss while
raising generated tokenizer-validity to 1.0.

I found one unfinished follow-up from the previous turn: a GPT-2-small runner
had been added and put into the default run path, but no completed job had
verified GPT-2-small scores. I also found that the scaled bigram experiments in
`reference.json` reused paper GPT-2 absolute losses as if they were direct
targets, making the reproduced scaled losses appear about 10x off.

## Issues Found

- **Critical:** the default `scripts/run.sh` included an unverified
  GPT-2-small experiment, while `scores.json` and the completed milestone only
  covered the two scaled bigram experiments.
- **Critical:** `reference.json` listed paper GPT-2 absolute bits/string values
  under the scaled bigram experiments. Those absolute values are not comparable
  to the CPU-safe bigram reproduction and created misleading score comparisons.
- **Minor:** method-agnostic target/evaluation docs described the unverified
  GPT-2-small follow-up as a current target.
- **Minor:** `README.md`, `manifest.json`, and `trajectory/` are read-only
  system bind mounts in this session, so Git initially reported them as dirty
  even though their committed content was not changed.

## Fixes Applied

- Removed the unverified GPT-2-small command from `scripts/run.sh`; the optional
  implementation remains in `method/gpt2_conditioning.py` for future extension.
- Updated `scoring/reference.json` so scaled bigram experiments keep only
  paper-derived relative log-loss context. Absolute scaled losses and constraint
  metrics are produced by `scores.json`, not treated as paper-reported targets.
- Cleaned `briefing/evaluation.md`, `scoring/TARGETS.md`,
  `scoring/EXPERIMENTS.md`, and `scoring/CONSTRAINTS.md` so from-scratch agents
  see only the active scored gym surface and high-level constraints.
- Updated `MILESTONES.md` and `PROGRESS.md` to record that GPT-2-small is
  optional/unverified and not part of the current milestone claim.
- Verified all URLs in `scripts/download.sh` with `curl -I -L --max-time 15`;
  every URL returned HTTP 200.
- Re-ran `bash scripts/evaluate.sh artifacts/canonical_conditioning`,
  `bash scripts/evaluate_train.sh artifacts/canonical_conditioning`,
  `bash scripts/evaluate_test.sh artifacts/canonical_conditioning`,
  `python validate.py --compare`, `bash -n scripts/*.sh`, and Python
  compilation checks. All passed.

## Final Judgment

The workspace is ready for gym use at `core_claim_plus`. Improve-mode agents
can run the packaged method and compare against the two-corpus scaled scores.
From-scratch agents can use the problem/data/evaluation contract without
importing or reading `method/`; `eval/` reads method artifacts and does not
import from `method/`.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: medium
- Key limitation: the verified gym scores use a scaled token bigram model and
  corpus subsets rather than the paper's full neural-LM experiments.
