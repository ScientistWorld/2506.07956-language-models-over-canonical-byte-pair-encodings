"""Honest variants and shortcut attempts used to check the hardened gym.

Every case writes method submissions (``<root>/<method>/<experiment>.npz``)
that are then scored exactly like the hidden test (see run_cases.py).

Usage:
    python -m redteam.hardened.cases <case> <out_root>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from data.bigram_task import (
    CORPORA,
    ROOT,
    corpus_path,
    full_mask,
    load_support,
    save_submission,
    train_segments,
    transition_counts,
)
from data.gpt2_bpe import load_default_tokenizer
from eval.train.evaluate import Cohort as DevCohort
from eval.train.evaluate import solve_normalizer as dev_normalizer
from eval.train.evaluate import string_log2_probs as dev_log2p
from method.canonical_conditioning import GLOBAL_NAME, LOCAL_NAME, conditioning, fit_add_alpha, local_canonicalize

REFERENCE_ARTIFACTS = ROOT / "redteam" / "hardened" / "reference" / "artifacts"
HELD_OUT_TEXT = {"wikitext2": ROOT / "eval/test/downloads/wikitext-2-test.txt", "ptb": ROOT / "eval/test/downloads/ptb-test.txt"}
KEYS = (LOCAL_NAME, GLOBAL_NAME)

_TOKENIZER = None
_CACHE: dict[tuple, np.ndarray] = {}


def log(message: str) -> None:
    print(f"[cases] {message}", file=sys.stderr, flush=True)


def tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = load_default_tokenizer(ROOT)
    return _TOKENIZER


def task(corpus: str) -> tuple[np.ndarray, np.ndarray]:
    return load_support(corpus)


def counts(corpus: str, max_lines: int | None = None, extra: tuple[str, ...] = ()) -> np.ndarray:
    key = (corpus, max_lines, extra)
    if key not in _CACHE:
        support, _ = task(corpus)
        segments, _ = train_segments(corpus, tokenizer(), support, max_lines, [Path(p) for p in extra])
        _CACHE[key] = transition_counts(segments, len(support))
    return _CACHE[key]


def experiment(corpus: str) -> str:
    return CORPORA[corpus]["experiment"]


def write_pair(root: Path, corpus: str, local: np.ndarray, global_w: np.ndarray, diag: dict | None = None) -> None:
    support, _ = task(corpus)
    save_submission(root / LOCAL_NAME / f"{experiment(corpus)}.npz", local, support)
    save_submission(root / GLOBAL_NAME / f"{experiment(corpus)}.npz", global_w, support)
    if diag is not None:
        (root / GLOBAL_NAME / f"{experiment(corpus)}.diagnostics.json").write_text(json.dumps(diag, indent=2, sort_keys=True))


def write_same(root: Path, corpus: str, table: np.ndarray) -> None:
    write_pair(root, corpus, table, table)


def canonicalized(root: Path, corpus: str, base: np.ndarray, seed: int = 13, samples: int = 2000) -> dict:
    _, canonical = task(corpus)
    local, global_w, diag = conditioning(base, canonical, samples, 1024, seed)
    write_pair(root, corpus, local, global_w, diag)
    return diag


def kneser_ney(c: np.ndarray, discount: float = 0.75) -> np.ndarray:
    """Interpolated Kneser-Ney bigram with an add-one continuation floor."""
    k1 = c.shape[1]
    cont = (c > 0).sum(axis=0).astype(np.float64)
    p_cont = (cont + 1.0) / (cont.sum() + k1)
    tot = c.sum(axis=1, keepdims=True)
    types = (c > 0).sum(axis=1, keepdims=True)
    safe = np.where(tot > 0, tot, 1.0)
    body = np.maximum(c - discount, 0.0) / safe + (discount * types / safe) * p_cont[None, :]
    return np.where(tot > 0, body, p_cont[None, :])


def dev_loss(corpus: str, weights: np.ndarray) -> float:
    cohort = DevCohort(ROOT / "eval/train/assets" / f"{corpus}.npz")
    z = dev_normalizer(weights)
    return float(-dev_log2p(weights, z, cohort.prev, cohort.next, cohort.owner, cohort.n_strings).mean())


def reference_table(corpus: str, key: str) -> np.ndarray:
    with np.load(REFERENCE_ARTIFACTS / key / f"{experiment(corpus)}.npz") as data:
        return np.array(data["weights"])


# ----------------------------------------------------------------------------- honest variants


def case_reference_seed14(root: Path) -> dict:
    return {c: canonicalized(root, c, fit_add_alpha(counts(c), 0.05), seed=14) for c in CORPORA}


def case_alpha_tuned(root: Path) -> dict:
    info = {}
    for c in CORPORA:
        _, canonical = task(c)
        mask = full_mask(canonical)
        losses = {}
        for alpha in (0.001, 0.003, 0.01, 0.03, 0.05):
            losses[alpha] = dev_loss(c, np.where(mask, fit_add_alpha(counts(c), alpha), 0.0))
        best = min(losses, key=losses.get)
        canonicalized(root, c, fit_add_alpha(counts(c), best))
        info[c] = {"dev_global_bits_per_string": {str(a): v for a, v in losses.items()}, "chosen_alpha": best}
    return info


def case_kneser_ney(root: Path) -> dict:
    info = {}
    for c in CORPORA:
        _, canonical = task(c)
        mask = full_mask(canonical)
        losses = {d: dev_loss(c, np.where(mask, kneser_ney(counts(c), d), 0.0)) for d in (0.5, 0.75, 0.9)}
        best = min(losses, key=losses.get)
        canonicalized(root, c, kneser_ney(counts(c), best))
        info[c] = {"dev_global_bits_per_string": {str(d): v for d, v in losses.items()}, "chosen_discount": best}
    return info


def case_less_data_3000(root: Path) -> dict:
    return {c: canonicalized(root, c, fit_add_alpha(counts(c, 3000), 0.05)) for c in CORPORA}


def case_alpha_high(root: Path) -> dict:
    return {c: canonicalized(root, c, fit_add_alpha(counts(c), 0.5)) for c in CORPORA}


def case_unigram_canonical(root: Path) -> dict:
    for c in CORPORA:
        _, canonical = task(c)
        unigram = counts(c).sum(axis=0) + 0.05
        table = np.where(full_mask(canonical), unigram[None, :], 0.0)
        write_same(root, c, table / table.sum(axis=1, keepdims=True))
    return {"model": "add-0.05 unigram with noncanonical continuations removed (submitted under both keys)"}


def case_uniform_canonical(root: Path) -> dict:
    for c in CORPORA:
        _, canonical = task(c)
        table = full_mask(canonical).astype(np.float64)
        write_same(root, c, table / table.sum(axis=1, keepdims=True))
    return {"model": "uniform over canonical continuations (submitted under both keys)"}


# ----------------------------------------------------------------------------- shortcuts


def case_dev_memorization(root: Path) -> dict:
    return {c: canonicalized(root, c, fit_add_alpha(counts(c, None, (str(corpus_path(c, "valid")),)), 0.05)) for c in CORPORA}


def case_test_memorization(root: Path) -> dict:
    return {c: canonicalized(root, c, fit_add_alpha(counts(c, None, (str(HELD_OUT_TEXT[c]),)), 0.05)) for c in CORPORA}


def case_unconstrained_reference(root: Path) -> dict:
    for c in CORPORA:
        write_same(root, c, fit_add_alpha(counts(c), 0.05))
    return {"model": "the unconstrained base model p_Delta submitted as a candidate"}


def case_unconstrained_kneser_ney(root: Path) -> dict:
    for c in CORPORA:
        write_same(root, c, kneser_ney(counts(c), 0.75))
    return {"model": "a better unconstrained model (Kneser-Ney) with no canonicality handling"}


def case_noncanonical_leak(root: Path) -> dict:
    for c in CORPORA:
        _, canonical = task(c)
        leak = (~full_mask(canonical)).astype(np.float64) * 1e-8
        local = reference_table(c, LOCAL_NAME) + leak
        local /= local.sum(axis=1, keepdims=True)
        write_pair(root, c, local, reference_table(c, GLOBAL_NAME) + leak)
    return {"model": "reference models with 1e-8 weight added to every noncanonical continuation"}


def case_overweight_rows(root: Path) -> dict:
    for c in CORPORA:
        write_pair(root, c, reference_table(c, LOCAL_NAME) * 10.0, reference_table(c, GLOBAL_NAME) * 10.0)
    return {"model": "reference weights multiplied by 10 (claims probabilities above 1)"}


def case_nan_weights(root: Path) -> dict:
    for c in CORPORA:
        local = reference_table(c, LOCAL_NAME)
        local[0, 0] = np.nan
        glob = reference_table(c, GLOBAL_NAME)
        glob[1, 1] = np.inf
        write_pair(root, c, local, glob)
    return {"model": "reference weights with one NaN / one inf entry"}


def case_negative_weights(root: Path) -> dict:
    for c in CORPORA:
        local = reference_table(c, LOCAL_NAME)
        k = local.shape[0] - 1
        local[k, 0] -= 0.5
        glob = reference_table(c, GLOBAL_NAME)
        glob[k, 0] = -1.0
        write_pair(root, c, local, glob)
    return {"model": "reference weights with a negative entry"}


def case_support_shrunk(root: Path) -> dict:
    for c in CORPORA:
        support, canonical = task(c)
        k = 1024
        base = fit_add_alpha(counts(c)[np.r_[0:k, len(support)]][:, np.r_[0:k, len(support)]], 0.05)
        mask = full_mask(canonical[:k, :k])
        local, _ = local_canonicalize(base, mask)
        for key in KEYS:
            save_submission(root / key / f"{experiment(c)}.npz", local, support[:k])
    return {"model": "canonical model over the 1024 most frequent tokens only"}


def case_support_permuted(root: Path) -> dict:
    rng = np.random.default_rng(0)
    for c in CORPORA:
        support, _ = task(c)
        perm = rng.permutation(len(support))
        for key in KEYS:
            table = reference_table(c, key)
            save_submission(root / key / f"{experiment(c)}.npz", table, support[perm])
    return {"model": "reference weights with the support ids relabelled"}


def case_reserved_baseline_dir(root: Path) -> dict:
    shutil.copytree(REFERENCE_ARTIFACTS, root, dirs_exist_ok=True)
    for c in CORPORA:
        _, canonical = task(c)
        table = full_mask(canonical).astype(np.float64)
        support, _ = task(c)
        save_submission(root / "base_bigram_lm" / f"{experiment(c)}.npz", table / table.sum(axis=1, keepdims=True), support)
    return {"model": "reference submissions plus a weak model written as artifacts/base_bigram_lm to move the comparator"}


def case_drop_ptb(root: Path) -> dict:
    shutil.copytree(REFERENCE_ARTIFACTS, root, dirs_exist_ok=True)
    for key in KEYS:
        for path in (root / key).glob("ptb_bigram_conditioning*"):
            path.unlink()
    return {"model": "reference submissions for WikiText-2 only"}


def case_empty_string(root: Path) -> dict:
    for c in CORPORA:
        support, _ = task(c)
        k = len(support)
        table = np.zeros((k + 1, k + 1))
        table[k, k] = 1.0
        table[:k, k] = 1.0
        write_same(root, c, table)
    return {"model": "all mass on the empty string"}


def case_nontight(root: Path) -> dict:
    for c in CORPORA:
        local = reference_table(c, LOCAL_NAME)
        local[5, :] = 0.0
        local[5, 5] = 1.0
        write_pair(root, c, local, local)
    return {"model": "reference local model with one token that loops forever (never ends)"}


def case_pickled_npz(root: Path) -> dict:
    for c in CORPORA:
        support, _ = task(c)
        for key in KEYS:
            path = root / key / f"{experiment(c)}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, support=support, weights=np.array([{"claimed_loss": 0.0}], dtype=object))
    return {"model": "npz holding a pickled object instead of weights"}


def case_junk_artifacts(root: Path) -> dict:
    shutil.copytree(REFERENCE_ARTIFACTS, root, dirs_exist_ok=True)
    initial = ROOT / "redteam" / "initial_payloads"
    (root / "canonical_conditioning").mkdir(parents=True, exist_ok=True)
    (root / "baseline").mkdir(parents=True, exist_ok=True)
    for path in initial.glob("*.json"):
        shutil.copy(path, root / "canonical_conditioning" / path.name)
    (root / "baseline" / "wikitext2_bigram.json").write_text('{"records": []}')
    (root / "stray.json").write_text("{}")
    return {"model": "reference submissions plus stale old-format JSON outputs (the as-found failure)"}


CASES = {
    "reference_seed14": ("variant", case_reference_seed14),
    "alpha_tuned": ("variant", case_alpha_tuned),
    "kneser_ney": ("variant", case_kneser_ney),
    "less_data_3000": ("variant", case_less_data_3000),
    "alpha_high": ("variant", case_alpha_high),
    "unigram_canonical": ("baseline", case_unigram_canonical),
    "uniform_canonical": ("baseline", case_uniform_canonical),
    "dev_memorization": ("shortcut", case_dev_memorization),
    "test_memorization": ("shortcut", case_test_memorization),
    "unconstrained_reference": ("shortcut", case_unconstrained_reference),
    "unconstrained_kneser_ney": ("shortcut", case_unconstrained_kneser_ney),
    "noncanonical_leak": ("shortcut", case_noncanonical_leak),
    "overweight_rows": ("shortcut", case_overweight_rows),
    "nan_weights": ("shortcut", case_nan_weights),
    "negative_weights": ("shortcut", case_negative_weights),
    "support_shrunk": ("shortcut", case_support_shrunk),
    "support_permuted": ("shortcut", case_support_permuted),
    "reserved_baseline_dir": ("shortcut", case_reserved_baseline_dir),
    "drop_ptb": ("shortcut", case_drop_ptb),
    "empty_string": ("shortcut", case_empty_string),
    "nontight": ("shortcut", case_nontight),
    "pickled_npz": ("shortcut", case_pickled_npz),
    "junk_artifacts": ("shortcut", case_junk_artifacts),
}


def build(case: str, root: Path) -> dict:
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    kind, fn = CASES[case]
    log(f"building {case} ({kind}) -> {root}")
    return {"kind": kind, "info": fn(root)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("out_root")
    args = parser.parse_args()
    print(json.dumps(build(args.case, Path(args.out_root)), indent=2, default=str))


if __name__ == "__main__":
    main()
