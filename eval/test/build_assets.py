"""Build the evaluator-owned hidden-test assets (maintainers only).

This is not run by the hidden test. For each corpus it writes
``eval/test/assets/<corpus>.npz`` containing:

* the fixed token support and canonical-bigram table (copied from
  ``data/support`` at build time, so later edits there cannot move the test);
* the held-out cohort: non-empty lines of the held-out test text whose GPT-2
  tokens all lie in the support and whose adjacent token pairs are canonical;
* the evaluator-owned baseline: an unconstrained add-0.05 token bigram fit on
  the full training file, scored exactly on the cohort;
* the failure floor: the uniform canonical model, scored exactly on the cohort.

Usage:
    python -m eval.test.build_assets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from data.bigram_task import (
    CORPORA,
    ROOT,
    SUPPORT_DIR,
    full_mask,
    load_support,
    read_lines,
    sha256_file,
    support_index,
    train_segments,
    transition_counts,
)
from data.gpt2_bpe import load_default_tokenizer
from eval.test.evaluate import ASSETS_DIR, canonical_mass, cohort_transitions, solve_normalizer, string_log2_probs

HELD_OUT_TEXT = {
    "wikitext2": "eval/test/downloads/wikitext-2-test.txt",
    "ptb": "eval/test/downloads/ptb-test.txt",
}
BASELINE_ALPHA = 0.05


def log(message: str) -> None:
    print(f"[build-test-assets] {message}", file=sys.stderr, flush=True)


def exact_scores(weights: np.ndarray, canonical: np.ndarray, flat: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, float]:
    k = canonical.shape[0]
    z = solve_normalizer(weights)
    prev, nxt, owner = cohort_transitions(flat, offsets, k)
    return string_log2_probs(weights, z, prev, nxt, owner, len(offsets) - 1), canonical_mass(weights, canonical, z)


def build(corpus: str, support_dir: Path, out_dir: Path, max_text_lines: int | None, max_train_lines: int | None) -> dict:
    tokenizer = load_default_tokenizer(ROOT)
    support, canonical = load_support(corpus, support_dir)
    k = len(support)
    index_of = support_index(support)
    text_path = ROOT / HELD_OUT_TEXT[corpus]
    lines = read_lines(text_path, max_text_lines)
    cohort: list[np.ndarray] = []
    out_of_support = noncanonical = 0
    for line in lines:
        seq = tokenizer.encode(line)
        if not seq:
            continue
        idx = [index_of.get(tok) for tok in seq]
        if any(i is None for i in idx):
            out_of_support += 1
            continue
        arr = np.asarray(idx, dtype=np.int64)
        if len(arr) > 1 and not bool(canonical[arr[:-1], arr[1:]].all()):
            noncanonical += 1
            continue
        cohort.append(arr)
    if not cohort:
        raise RuntimeError(f"{corpus}: empty held-out cohort")
    lengths = np.asarray([len(seq) for seq in cohort], dtype=np.int64)
    offsets = np.zeros(len(cohort) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    flat = np.concatenate(cohort)

    segments, train_lines = train_segments(corpus, tokenizer, support, max_train_lines)
    counts = transition_counts(segments, k)
    baseline = (counts + BASELINE_ALPHA) / (counts.sum(axis=1, keepdims=True) + BASELINE_ALPHA * (k + 1))
    base_log2p, base_mass = exact_scores(baseline, canonical, flat, offsets)
    uniform = full_mask(canonical).astype(np.float64)
    uniform /= uniform.sum(axis=1, keepdims=True)
    fail_log2p, fail_mass = exact_scores(uniform, canonical, flat, offsets)

    out_dir.mkdir(parents=True, exist_ok=True)
    asset_path = out_dir / f"{corpus}.npz"
    np.savez_compressed(
        asset_path,
        support=support,
        canonical_packed=np.packbits(canonical, axis=1),
        cohort_tokens=flat.astype(np.int32),
        cohort_offsets=offsets,
        baseline_log2p=base_log2p,
        baseline_canonical_mass=np.asarray(base_mass),
        baseline_alpha=np.asarray(BASELINE_ALPHA),
        failure_log2p=fail_log2p,
        failure_canonical_mass=np.asarray(fail_mass),
    )
    entry = {
        "experiment": CORPORA[corpus]["experiment"],
        "held_out_text": HELD_OUT_TEXT[corpus],
        "held_out_text_sha256": sha256_file(text_path),
        "held_out_lines": len(lines),
        "dropped_out_of_support": out_of_support,
        "dropped_noncanonical": noncanonical,
        "cohort_strings": len(cohort),
        "cohort_tokens": int(lengths.sum()),
        "mean_tokens_per_string": float(lengths.mean()),
        "baseline": f"unconstrained add-{BASELINE_ALPHA} token bigram on {train_lines} training lines",
        "baseline_bits_per_string": float(-base_log2p.mean()),
        "baseline_canonical_mass": base_mass,
        "failure_floor_bits_per_string": float(-fail_log2p.mean()),
        "support_table_sha256": sha256_file(Path(support_dir) / f"{corpus}_canonical.npz"),
        "asset_sha256": sha256_file(asset_path),
    }
    log(f"{corpus}: {json.dumps(entry, sort_keys=True)}")
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="all", choices=["all", *CORPORA])
    parser.add_argument("--support-dir", default=str(SUPPORT_DIR))
    parser.add_argument("--out-dir", default=str(ASSETS_DIR))
    parser.add_argument("--max-text-lines", type=int, default=None, help="smoke tests only")
    parser.add_argument("--max-train-lines", type=int, default=None, help="smoke tests only")
    args = parser.parse_args()
    corpora = list(CORPORA) if args.corpus == "all" else [args.corpus]
    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for corpus in corpora:
        manifest[corpus] = build(corpus, Path(args.support_dir), out_dir, args.max_text_lines, args.max_train_lines)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
