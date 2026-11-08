"""Canonicality by conditioning (Vieira et al., ICML 2025, Sec. 3) for a token bigram LM.

The base token LM p_Delta is an add-alpha bigram model over the fixed GPT-2 BPE
token support, fit on the full training file. On top of it the method builds:

* the locally canonicalized model ell (Def. 2, Eq. 9): at each step every next
  token whose bigram with the previous token is noncanonical (BPE bigram test,
  App. B / Theorem 1) is masked out and the next-token distribution is
  renormalized by the local normalizer;
* the globally canonicalized model g (Def. 1, Eq. 6): p_Delta conditioned on
  the token string being canonical, g = p_Delta * 1{canonical} / Z. Following
  Sec. 3.2.3, the canonicality rate Z is estimated by importance sampling with
  M samples from ell (Prop. 4, Eq. 12-13) whose weights are the products of
  local normalizers (Prop. 2), and importance resampling (Eq. 14) draws
  approximate samples from g.

Outputs per corpus experiment:

* ``<out>/local_canonical_bigram_lm/<experiment>.npz``: the locally normalized
  transition table of ell;
* ``<out>/global_canonical_bigram_lm/<experiment>.npz``: p_Delta restricted to
  canonical bigrams (unnormalized; normalizing it over all strings is exactly
  the conditioning that defines g);
* ``<out>/global_canonical_bigram_lm/<experiment>.diagnostics.json``: the
  importance-sampling estimate of log Z, the exact log Z of this bigram model,
  and sample canonicality rates. The evaluator never reads diagnostics.

Usage:
    python -m method.canonical_conditioning --corpus all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from data.bigram_task import (
    CORPORA,
    ROOT,
    SUPPORT_DIR,
    exact_log2_normalizer,
    full_mask,
    load_support,
    save_submission,
    train_segments,
    transition_counts,
)
from data.gpt2_bpe import load_default_tokenizer

LOCAL_NAME = "local_canonical_bigram_lm"
GLOBAL_NAME = "global_canonical_bigram_lm"


def log(message: str) -> None:
    print(f"[canonical-conditioning] {message}", file=sys.stderr, flush=True)


def fit_add_alpha(counts: np.ndarray, alpha: float) -> np.ndarray:
    """Unconstrained token LM p_Delta: add-alpha smoothed next-symbol distributions."""
    k1 = counts.shape[1]
    return (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * k1)


def local_canonicalize(base: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eq. 9: mask noncanonical continuations and renormalize each next-symbol distribution."""
    masked = np.where(mask, base, 0.0)
    norms = masked.sum(axis=1)
    return masked / norms[:, None], norms


def sample_strings(
    table: np.ndarray,
    n: int,
    rng: np.random.Generator,
    max_tokens: int,
    log2_norms: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Ancestral sampling from a (row-normalized) transition table.

    Returns the sampled strings (support indices), the accumulated log2 of the
    local normalizers along each path (Prop. 2 weights when ``log2_norms`` is
    given), and a flag for strings that hit ``max_tokens`` without ending.
    """
    k = table.shape[0] - 1
    cum = np.cumsum(table, axis=1)
    state = np.full(n, k, dtype=np.int64)
    active = np.ones(n, dtype=bool)
    truncated = np.zeros(n, dtype=bool)
    log2w = np.zeros(n, dtype=np.float64)
    tokens: list[list[int]] = [[] for _ in range(n)]
    for step in range(max_tokens + 1):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        rows = cum[state[idx]]
        u = (1.0 - rng.random(idx.size)) * rows[:, -1]
        nxt = np.minimum((rows < u[:, None]).sum(axis=1), k)
        if log2_norms is not None:
            log2w[idx] += log2_norms[state[idx]]
        ended = nxt == k
        if step == max_tokens:
            truncated[idx[~ended]] = True
            active[idx] = False
            break
        for j, tok in zip(idx[~ended].tolist(), nxt[~ended].tolist()):
            tokens[j].append(tok)
        state[idx] = nxt
        active[idx[ended]] = False
    return [np.asarray(seq, dtype=np.int64) for seq in tokens], log2w, truncated


def canonical_rate(strings: list[np.ndarray], canonical: np.ndarray) -> float:
    ok = [bool(canonical[s[:-1], s[1:]].all()) if len(s) > 1 else True for s in strings]
    return float(np.mean(ok)) if ok else float("nan")


def log2_mean_exp2(values: np.ndarray) -> float:
    m = float(values.max())
    return m + math.log2(float(np.mean(np.exp2(values - m))))


def conditioning(
    base: np.ndarray,
    canonical: np.ndarray,
    samples: int,
    max_tokens: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (local table, global weights, diagnostics) for a base token LM."""
    mask = full_mask(canonical)
    local, norms = local_canonicalize(base, mask)
    global_weights = np.where(mask, base, 0.0)
    rng = np.random.default_rng(seed)

    base_strings, _, base_trunc = sample_strings(base, samples, rng, max_tokens)
    local_strings, log2w, local_trunc = sample_strings(local, samples, rng, max_tokens, np.log2(norms))
    complete = ~local_trunc
    if not complete.any():
        raise RuntimeError("no complete samples from the local model")
    log2_z_hat = log2_mean_exp2(log2w[complete])
    weights = np.exp2(log2w[complete] - log2w[complete].max())
    picks = rng.choice(np.nonzero(complete)[0], size=samples, p=weights / weights.sum())
    resampled = [local_strings[i] for i in picks]
    log2_z_exact, _ = exact_log2_normalizer(global_weights)
    diagnostics = {
        "samples": samples,
        "sample_max_tokens": max_tokens,
        "seed": seed,
        "local_truncated_samples": int(local_trunc.sum()),
        "base_truncated_samples": int(base_trunc.sum()),
        "log2_Z_importance_sampling": log2_z_hat,
        "log2_Z_exact": log2_z_exact,
        "importance_weight_ess": float(weights.sum() ** 2 / (weights**2).sum()),
        "mean_local_sample_tokens": float(np.mean([len(s) for s in local_strings])),
        "base_sample_canonical_rate": canonical_rate(base_strings, canonical),
        "local_sample_canonical_rate": canonical_rate(local_strings, canonical),
        "global_resample_canonical_rate": canonical_rate(resampled, canonical),
        "min_local_normalizer": float(norms.min()),
    }
    return local, global_weights, diagnostics


def run_corpus(corpus: str, args: argparse.Namespace, tokenizer) -> dict:
    experiment = CORPORA[corpus]["experiment"]
    support, canonical = load_support(corpus, args.support_dir)
    k = len(support)
    segments, n_lines = train_segments(corpus, tokenizer, support, args.max_train_lines)
    counts = transition_counts(segments, k)
    log(f"{corpus}: fit add-{args.alpha} bigram on {n_lines} lines, {len(segments)} segments, {int(counts.sum())} transitions")
    base = fit_add_alpha(counts, args.alpha)
    local, global_weights, diagnostics = conditioning(base, canonical, args.samples, args.sample_max_tokens, args.seed)
    out_root = Path(args.out_root)
    save_submission(out_root / args.local_name / f"{experiment}.npz", local, support)
    save_submission(out_root / args.global_name / f"{experiment}.npz", global_weights, support)
    diagnostics.update(
        {
            "experiment": experiment,
            "alpha": args.alpha,
            "train_lines": n_lines,
            "train_segments": len(segments),
            "train_transitions": int(counts.sum()),
            "support_size": k,
            "note": "diagnostics only; the evaluator scores the submitted weights exactly",
        }
    )
    diag_path = out_root / args.global_name / f"{experiment}.diagnostics.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    log(
        f"{corpus}: log2 Z IS={diagnostics['log2_Z_importance_sampling']:.6f} exact={diagnostics['log2_Z_exact']:.6f}; "
        f"canonical rate base={diagnostics['base_sample_canonical_rate']:.3f} "
        f"local={diagnostics['local_sample_canonical_rate']:.3f} global={diagnostics['global_resample_canonical_rate']:.3f}"
    )
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="all", choices=["all", *CORPORA])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-train-lines", type=int, default=None, help="default: the full training file")
    parser.add_argument("--samples", type=int, default=2000, help="M samples from ell for the IS estimate (paper: 2000)")
    parser.add_argument("--sample-max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out-root", default=str(ROOT / "artifacts"))
    parser.add_argument("--support-dir", default=str(SUPPORT_DIR))
    parser.add_argument("--local-name", default=LOCAL_NAME)
    parser.add_argument("--global-name", default=GLOBAL_NAME)
    args = parser.parse_args()
    tokenizer = load_default_tokenizer(ROOT)
    corpora = list(CORPORA) if args.corpus == "all" else [args.corpus]
    for corpus in corpora:
        run_corpus(corpus, args, tokenizer)


if __name__ == "__main__":
    main()
