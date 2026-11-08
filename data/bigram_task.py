"""Shared task definition for the canonical token-bigram language-model benchmark.

The benchmark fixes, for each corpus:

* the GPT-2 byte-level BPE tokenizer (``data/downloads/gpt2-tokenizer``),
* a token support V of K GPT-2 token ids (``data/support/<corpus>.json``),
* the table of canonical token bigrams over V x V
  (``data/support/<corpus>_canonical.npz``).

A submission is a weighted first-order Markov model over V, stored as an
``.npz`` file with exactly two arrays:

``support``  int64[K]          -- must equal the benchmark support, same order.
``weights``  float64[K+1, K+1] -- ``weights[i, j]`` is the weight of moving
                                  from symbol i to symbol j.  Indices 0..K-1
                                  are support tokens, row K is the start of the
                                  string and column K is the end of the string.

A token string t_1..t_n receives weight
``W[K, t_1] * W[t_1, t_2] * ... * W[t_n, K]``.  The evaluator normalizes these
weights exactly over all finite strings, so rows may be sub-stochastic; every
row must be non-negative and sum to at most 1.

This module only contains data/format helpers. It does not implement any
modelling method.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from data.wikitext import read_wikitext_lines

ROOT = Path(__file__).resolve().parents[1]
SUPPORT_DIR = ROOT / "data" / "support"
ROW_SUM_TOLERANCE = 1e-6

CORPORA: dict[str, dict[str, str]] = {
    "wikitext2": {
        "experiment": "wikitext2_bigram_conditioning",
        "train": "data/downloads/wikitext-2-train.txt",
        "valid": "data/downloads/wikitext-2-valid.txt",
    },
    "ptb": {
        "experiment": "ptb_bigram_conditioning",
        "train": "data/downloads/ptb-train.txt",
        "valid": "data/downloads/ptb-valid.txt",
    },
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_path(corpus: str, split: str) -> Path:
    return ROOT / CORPORA[corpus][split]


def read_lines(path: str | Path, max_lines: int | None = None) -> list[str]:
    return read_wikitext_lines(path, max_lines)


def load_support(corpus: str, support_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (support token ids int64[K], canonical bool[K, K])."""
    support_dir = Path(support_dir) if support_dir is not None else SUPPORT_DIR
    meta = json.loads((support_dir / f"{corpus}.json").read_text())
    support = np.asarray(meta["support"], dtype=np.int64)
    with np.load(support_dir / f"{corpus}_canonical.npz", allow_pickle=False) as data:
        packed = data["canonical_packed"]
        k = int(data["support_size"])
    canonical = np.unpackbits(packed, axis=1, count=k).astype(bool)
    if canonical.shape != (len(support), len(support)):
        raise ValueError(f"canonical table shape {canonical.shape} does not match support size {len(support)}")
    return support, canonical


def support_index(support: np.ndarray) -> dict[int, int]:
    return {int(tok): i for i, tok in enumerate(support)}


def split_segments(sequences: list[list[int]], index_of: dict[int, int]) -> list[np.ndarray]:
    """Split token sequences into maximal runs of in-support tokens (as support indices)."""
    segments: list[np.ndarray] = []
    for seq in sequences:
        current: list[int] = []
        for tok in seq:
            idx = index_of.get(tok)
            if idx is None:
                if current:
                    segments.append(np.asarray(current, dtype=np.int64))
                    current = []
            else:
                current.append(idx)
        if current:
            segments.append(np.asarray(current, dtype=np.int64))
    return segments


def train_segments(
    corpus: str,
    tokenizer,
    support: np.ndarray,
    max_lines: int | None = None,
    extra_paths: list[str | Path] | None = None,
) -> tuple[list[np.ndarray], int]:
    """Tokenize the training file and return in-support segments and the line count."""
    lines = read_lines(corpus_path(corpus, "train"), max_lines)
    for path in extra_paths or []:
        lines += read_lines(path)
    sequences = [tokenizer.encode(line) for line in lines]
    return split_segments(sequences, support_index(support)), len(lines)


def transition_counts(segments: list[np.ndarray], k: int) -> np.ndarray:
    """Counts of (previous, next) symbols; row k is the string start, column k the string end."""
    counts = np.zeros((k + 1, k + 1), dtype=np.float64)
    for seg in segments:
        if len(seg) == 0:
            continue
        prev = np.concatenate(([k], seg))
        nxt = np.concatenate((seg, [k]))
        np.add.at(counts, (prev, nxt), 1.0)
    return counts


def full_mask(canonical: np.ndarray) -> np.ndarray:
    """Extend the token-token canonical table with always-allowed start row and end column."""
    k = canonical.shape[0]
    mask = np.ones((k + 1, k + 1), dtype=bool)
    mask[:k, :k] = canonical
    return mask


def save_submission(path: str | Path, weights: np.ndarray, support: np.ndarray) -> Path:
    weights = np.asarray(weights, dtype=np.float64)
    support = np.asarray(support, dtype=np.int64)
    k = len(support)
    if weights.shape != (k + 1, k + 1):
        raise ValueError(f"weights must have shape {(k + 1, k + 1)}, got {weights.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, support=support, weights=weights)
    return path


def exact_log2_normalizer(weights: np.ndarray) -> tuple[float, float]:
    """Return (log2 Z, max expected visits) for a weighted bigram model.

    Z sums the weights of all finite strings. Raises ValueError when the
    weights do not define a convergent (terminating) model.
    """
    k = weights.shape[0] - 1
    a = weights[:k, :k]
    rhs = np.stack([weights[:k, k], np.ones(k)], axis=1)
    system = np.eye(k) - a
    solution = np.linalg.solve(system, rhs)
    beta, visits = solution[:, 0], solution[:, 1]
    if not (np.all(np.isfinite(solution)) and visits.min() >= 1.0 - 1e-9):
        raise ValueError("weights do not define a terminating model")
    z = float(weights[k, k] + weights[k, :k] @ beta)
    if not (z > 0.0 and np.isfinite(z)):
        raise ValueError(f"invalid normalizer {z}")
    return float(np.log2(z)), float(visits.max())
