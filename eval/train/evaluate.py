"""Visible development (train-side) scorer for canonical token-bigram language models.

Each method on disk lives in ``artifacts/<method>/<experiment>.npz`` and holds a
weighted first-order Markov model over the fixed token support (format in
``briefing/evaluation.md``). For every experiment this scorer:

1. checks that the submission uses the fixed support and holds finite,
   non-negative weights whose rows sum to at most 1;
2. normalizes the weights exactly over all finite strings with a linear solve
   and certifies that the model terminates;
3. computes exact log-probabilities of the development strings, drawn from
   the visible validation split;
4. computes the exact probability that a string drawn from the model is
   canonical (every adjacent token pair canonical) and requires it to be 1.

A submission that fails a check, or is missing for an experiment, is scored at
the failure floor: the development loss of the uniform canonical model. The
baseline row ``base_bigram_lm`` is owned by this evaluator. Nothing a method
writes besides its weights is read, and no method code is imported.

This file is self-contained (standard library + numpy) and is meant to be run
as ``python -I eval/train/evaluate.py``.
It mirrors the hidden test scorer but shares no code with it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SLICE = "train"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS_DIR = HERE / "assets"
EXPERIMENTS = {
    "wikitext2_bigram_conditioning": "wikitext2",
    "ptb_bigram_conditioning": "ptb",
}
BASELINE_NAME = "base_bigram_lm"
ROW_SUM_TOLERANCE = 1e-6
CANONICAL_TOLERANCE = 1e-9
MAX_EXPECTED_VISITS = 1e6
RESIDUAL_TOLERANCE = 1e-8
METRICS = (
    "relative_log_loss_to_baseline",
    "log_loss_bits_per_string",
    "loss_improvement_bits_per_string",
    "log_loss_bits_per_token",
    "canonical_sample_rate",
)


class InvalidSubmission(Exception):
    def __init__(self, reason: str, canonical_mass: float = 0.0):
        super().__init__(reason)
        self.reason = reason
        self.canonical_mass = canonical_mass


def cohort_transitions(flat: np.ndarray, offsets: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (previous symbol, next symbol, string index) for every transition of every string."""
    prev, nxt, owner = [], [], []
    for i in range(len(offsets) - 1):
        toks = flat[offsets[i] : offsets[i + 1]]
        prev.append(np.concatenate(([k], toks)))
        nxt.append(np.concatenate((toks, [k])))
        owner.append(np.full(len(toks) + 1, i, dtype=np.int64))
    return np.concatenate(prev).astype(np.int64), np.concatenate(nxt).astype(np.int64), np.concatenate(owner)


def solve_normalizer(weights: np.ndarray) -> float:
    """Exact total weight Z of all finite strings; raises if the model does not terminate."""
    k = weights.shape[0] - 1
    system = np.eye(k) - weights[:k, :k]
    rhs = np.stack([weights[:k, k], np.ones(k)], axis=1)
    try:
        solution = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError as exc:
        raise InvalidSubmission(f"weights do not define a terminating model ({exc})") from exc
    if not np.all(np.isfinite(solution)):
        raise InvalidSubmission("weights do not define a terminating model (non-finite solve)")
    residual = float(np.abs(system @ solution - rhs).max())
    if residual > RESIDUAL_TOLERANCE * max(1.0, float(np.abs(solution).max())):
        raise InvalidSubmission(f"normalizer solve is numerically unreliable (residual {residual:.3g})")
    beta, visits = solution[:, 0], solution[:, 1]
    # For non-negative A, a solution of (I - A) x = 1 with x >= 1 certifies spectral radius < 1.
    if visits.min() < 1.0 - 1e-9:
        raise InvalidSubmission("weights do not define a terminating model (spectral radius >= 1)")
    if visits.max() > MAX_EXPECTED_VISITS:
        raise InvalidSubmission(f"expected string length exceeds {MAX_EXPECTED_VISITS:g} tokens")
    if beta.min() < -1e-12:
        raise InvalidSubmission("negative termination mass")
    z = float(weights[k, k] + weights[k, :k] @ beta)
    if not (np.isfinite(z) and z > 0.0):
        raise InvalidSubmission(f"invalid normalizer {z}")
    return z


def canonical_mass(weights: np.ndarray, canonical: np.ndarray, z: float) -> float:
    """Exact probability that a string drawn from the normalized model is canonical."""
    k = canonical.shape[0]
    tokens = weights[:k, :k]
    if not np.any(tokens[~canonical] > 0.0):
        return 1.0
    masked = weights.copy()
    masked[:k, :k] = np.where(canonical, tokens, 0.0)
    return float(min(1.0, solve_normalizer(masked) / z))


def string_log2_probs(weights: np.ndarray, z: float, prev: np.ndarray, nxt: np.ndarray, owner: np.ndarray, n: int) -> np.ndarray:
    values = weights[prev, nxt]
    if np.any(values <= 0.0):
        bad = int(np.unique(owner[values <= 0.0]).size)
        raise InvalidSubmission(f"zero probability for {bad} development string(s)")
    log2w = np.bincount(owner, weights=np.log2(values), minlength=n)
    return log2w - np.log2(z)


def check_weights(support: np.ndarray, weights: np.ndarray, expected_support: np.ndarray) -> np.ndarray:
    k = len(expected_support)
    if support.dtype.kind not in "iu" or support.shape != (k,) or not np.array_equal(support.astype(np.int64), expected_support):
        raise InvalidSubmission("support does not match the fixed benchmark support")
    if weights.dtype.kind != "f" or weights.shape != (k + 1, k + 1):
        raise InvalidSubmission(f"weights must be a float array of shape {(k + 1, k + 1)}")
    weights = weights.astype(np.float64)
    if not np.all(np.isfinite(weights)):
        raise InvalidSubmission("weights contain non-finite values")
    if weights.min() < 0.0:
        raise InvalidSubmission("weights contain negative values")
    row_sums = weights.sum(axis=1)
    if row_sums.max() > 1.0 + ROW_SUM_TOLERANCE:
        raise InvalidSubmission(f"a row of weights sums to {row_sums.max():.6g} > 1")
    return weights


class Cohort:
    """Evaluator-owned assets for one corpus: support, canonical table, development strings, baseline."""

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as data:
            self.support = data["support"].astype(np.int64)
            self.k = len(self.support)
            self.canonical = np.unpackbits(data["canonical_packed"], axis=1, count=self.k).astype(bool)
            flat = data["cohort_tokens"].astype(np.int64)
            offsets = data["cohort_offsets"].astype(np.int64)
            self.baseline_log2p = data["baseline_log2p"].astype(np.float64)
            self.baseline_canonical_mass = float(data["baseline_canonical_mass"])
            self.failure_log2p = data["failure_log2p"].astype(np.float64)
        self.n_strings = len(offsets) - 1
        self.n_tokens = int(offsets[-1])
        self.prev, self.next, self.owner = cohort_transitions(flat, offsets, self.k)
        self.baseline_loss = float(-self.baseline_log2p.mean())

    def metrics(self, log2p: np.ndarray, mass: float) -> dict[str, float]:
        loss = float(-log2p.mean())
        return {
            "relative_log_loss_to_baseline": loss / self.baseline_loss,
            "log_loss_bits_per_string": loss,
            "loss_improvement_bits_per_string": self.baseline_loss - loss,
            "log_loss_bits_per_token": float(-log2p.sum() / self.n_tokens),
            "canonical_sample_rate": float(mass),
        }

    def score(self, path: Path) -> tuple[np.ndarray, float, dict]:
        try:
            with np.load(path, allow_pickle=False) as data:
                if "support" not in data.files or "weights" not in data.files:
                    raise InvalidSubmission("submission must contain 'support' and 'weights' arrays")
                support = np.array(data["support"])
                weights = np.array(data["weights"])
        except InvalidSubmission:
            raise
        except Exception as exc:  # unreadable or pickled content
            raise InvalidSubmission(f"cannot read submission: {type(exc).__name__}: {exc}") from exc
        weights = check_weights(support, weights, self.support)
        z = solve_normalizer(weights)
        mass = canonical_mass(weights, self.canonical, z)
        if 1.0 - mass > CANONICAL_TOLERANCE:
            raise InvalidSubmission(f"probability of noncanonical strings is {1.0 - mass:.3g}", canonical_mass=mass)
        log2p = string_log2_probs(weights, z, self.prev, self.next, self.owner, self.n_strings)
        if log2p.max() > 1e-9:
            raise InvalidSubmission("a development probability exceeds 1 (numerical failure)", canonical_mass=mass)
        return log2p, mass, {"valid": True, "log2_normalizer": float(np.log2(z))}


def discover_methods(root: Path) -> tuple[dict[str, Path], list[str]]:
    ignored: list[str] = []
    if not root.is_dir():
        return {}, [f"{root} is not a directory"]
    if any((root / f"{exp}.npz").is_file() for exp in EXPERIMENTS):
        return {root.name: root}, ignored
    methods: dict[str, Path] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith((".", "_")):
            continue
        if not any((entry / f"{exp}.npz").is_file() for exp in EXPERIMENTS):
            ignored.append(f"{entry.name}: no <experiment>.npz submission")
            continue
        if entry.name == BASELINE_NAME:
            ignored.append(f"{entry.name}: reserved for the evaluator-owned baseline")
            continue
        methods[entry.name] = entry
    return methods, ignored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="?", default=str(ROOT / "artifacts"))
    parser.add_argument("--output", default=str(ROOT / "scoring" / "scores_train.json"))
    parser.add_argument("--assets-dir", default=str(ASSETS_DIR))
    parser.add_argument("--per-string-out", default=None, help="optional per-string log2 probabilities (analysis only)")
    args = parser.parse_args()

    cohorts = {exp: Cohort(Path(args.assets_dir) / f"{corpus}.npz") for exp, corpus in EXPERIMENTS.items()}
    methods, ignored = discover_methods(Path(args.artifacts))
    results: dict[str, dict[str, dict[str, float]]] = {}
    status: dict[str, dict[str, dict]] = {}
    per_string: dict[str, dict[str, list[float]]] = {}
    for exp, cohort in cohorts.items():
        results[exp] = {BASELINE_NAME: cohort.metrics(cohort.baseline_log2p, cohort.baseline_canonical_mass)}
        status[exp] = {BASELINE_NAME: {"valid": True, "owner": "evaluator"}}
        per_string[exp] = {BASELINE_NAME: cohort.baseline_log2p.tolist()}
        for name, directory in methods.items():
            path = directory / f"{exp}.npz"
            try:
                if not path.is_file():
                    raise InvalidSubmission("missing submission for this experiment")
                log2p, mass, info = cohort.score(path)
            except InvalidSubmission as exc:
                log2p, mass = cohort.failure_log2p, exc.canonical_mass
                info = {"valid": False, "reason": exc.reason, "scored_as": "failure floor (uniform canonical model)"}
            results[exp][name] = cohort.metrics(log2p, mass)
            status[exp][name] = info
            per_string[exp][name] = log2p.tolist()
            print(
                f"[{SLICE}] {exp} {name}: valid={info['valid']} "
                f"bits/string={results[exp][name]['log_loss_bits_per_string']:.4f} "
                f"canonical={mass:.6f}" + ("" if info["valid"] else f" reason={info['reason']}"),
                flush=True,
            )

    if not methods:
        print(f"[{SLICE}] warning: no method submissions found under {args.artifacts}", file=sys.stderr)
    payload = {
        "_metadata": {
            "slice": SLICE,
            "artifacts_root": str(args.artifacts),
            "cohort": {exp: {"strings": c.n_strings, "tokens": c.n_tokens} for exp, c in cohorts.items()},
            "status": status,
            "ignored": ignored,
            "failure_rule": "invalid or missing submissions are scored at the uniform canonical model's development loss",
        },
        "experiments": {exp: {"results": rows} for exp, rows in results.items()},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.per_string_out:
        Path(args.per_string_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.per_string_out).write_text(json.dumps(per_string))


if __name__ == "__main__":
    main()
