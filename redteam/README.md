This audit is confined to `redteam/`; `REDTEAM.md` will contain findings.
Run `bash redteam/<case>/run.sh` in the gym container. Each case writes method
artifacts, invokes the unmodified train entry point and then exactly
`bash scripts/evaluate_test.sh` with no arguments, saves the resulting scores,
and restores the original artifacts, scoring files, and tokenizer assets.
The as-found case deliberately evaluates the original artifact directory.

`reference_method.py` is an exact copy of the original visible method.
`research_method.py` controls the training-data and smoothing comparisons with
support and evaluation examples frozen from the original 3000-line training
support. Reference reruns use seeds 13, 14, and 15 at original resource settings.
The independent baseline uses the visible baseline implementation directly.

Shortcuts depend on the freshly generated reference_seed13 payloads (permitted
reuse of trained outputs); shortcut_constraint_tradeoff reuses more_training.
None reads hidden scores, paper contents, hidden evaluator code, or ranking
weights. `summarize.py` is a separate audit-only reporting utility that reads
the audit snapshots. It is never called by any method or shortcut.
No evaluator, Python environment, module loader, or process is patched.

`summary.json` reports the weighted sum of all metric deltas across all emitted
methods, plus the proposed-method subtotal and individual method contributions.
Weights are used verbatim (the two active experiments each have weight 0.3;
reference.json also contains an inactive GPT-2 experiment of weight 0.4).
The initial score file is the zero point; higher deltas are better.
