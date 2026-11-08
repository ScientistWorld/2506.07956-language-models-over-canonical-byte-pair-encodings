#!/bin/bash
# Hardening job: self-tests, rebuild the fixed task assets, rerun the reference
# end to end, run the simple baselines, then score honest variants and shortcut
# attempts exactly like the hidden test.
#
# The reference-only reproduction is the "reference" section below
# (scripts/method.sh + the three evaluators); the task assets it reads are
# committed (rebuild them with scripts/build_assets.sh).

set -e

cd /home/user
export PYTHONUNBUFFERED=1

echo "== self-test (exact scorer vs brute force, sampler, IS estimate)"
python -m redteam.hardened.selftest

echo "== smoke pipeline on a tiny support"
S=tmp/smoke
if [ ! -f "$S/scores_train.json" ]; then
rm -rf "$S"
python -m data.build_support --support-size 128 --max-train-lines 2000 --out-dir "$S/support"
python -m eval.train.build_assets --support-dir "$S/support" --out-dir "$S/train_assets" --max-train-lines 2000
python -m eval.test.build_assets --support-dir "$S/support" --out-dir "$S/test_assets" --max-train-lines 2000
python -m method.canonical_conditioning --support-dir "$S/support" --out-root "$S/artifacts" --max-train-lines 2000 --samples 200
python -m baseline.base_bigram --kind unconstrained --support-dir "$S/support" --out-root "$S/artifacts" --max-train-lines 2000
python -I eval/test/evaluate.py "$S/artifacts" --assets-dir "$S/test_assets" --output "$S/scores_test.json"
python -I eval/train/evaluate.py "$S/artifacts" --assets-dir "$S/train_assets" --output "$S/scores_train.json"
fi

echo "== fixed task assets (support, canonical bigrams, dev/test cohorts, evaluator baseline)"
# (skipped when a previous attempt of this job already built them)
[ -f eval/test/assets/manifest.json ] && [ -f data/support/ptb_canonical.npz ] || bash scripts/build_assets.sh

echo "== reference"
rm -rf artifacts
bash scripts/method.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python validate.py --post-design --compare || echo "validate reported problems (see above)"

echo "== simple baselines"
bash scripts/baseline.sh unigram
bash scripts/baseline.sh unconstrained

echo "== hardening cases (variants and shortcuts through the hidden test)"
python -m redteam.hardened.run_cases || echo "run_cases failed"

echo "== final reference scores"
bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python validate.py --post-design --compare
