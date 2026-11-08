"""Numerical self-test of the exact scorer and the reference sampler on tiny models.

Compares the evaluator's closed-form normalizer, canonical mass, and string
log-probabilities with brute-force enumeration, and checks that the method's
ancestral sampler and importance-sampling estimate of Z agree with the exact
values. Run: python -m redteam.hardened.selftest
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from eval.test import evaluate as test_eval
from eval.train import evaluate as train_eval
from method.canonical_conditioning import conditioning, sample_strings


def brute_force(weights: np.ndarray, canonical: np.ndarray, max_len: int) -> tuple[float, float, dict]:
    k = canonical.shape[0]
    total = canon = 0.0
    probs: dict[tuple[int, ...], float] = {}
    for n in range(max_len + 1):
        for seq in itertools.product(range(k), repeat=n):
            path = (k,) + seq + (k,)
            w = 1.0
            for a, b in zip(path, path[1:]):
                w *= weights[a, b]
            total += w
            if all(canonical[a, b] for a, b in zip(seq, seq[1:])):
                canon += w
            if n <= 3:
                probs[seq] = w
    return total, canon, probs


def random_model(rng: np.random.Generator, k: int, eos: float) -> np.ndarray:
    w = rng.random((k + 1, k + 1))
    w[:, k] = 0.0
    w = w / w.sum(axis=1, keepdims=True) * (1.0 - eos)
    w[:, k] = eos
    w[k, k] = 0.05
    w[k, :k] *= 0.95 / w[k, :k].sum()
    return w * rng.uniform(0.7, 1.0, size=(k + 1, 1))


def main() -> None:
    rng = np.random.default_rng(0)
    k = 3
    canonical = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=bool)
    for module in (test_eval, train_eval):
        weights = random_model(rng, k, eos=0.65)
        z = module.solve_normalizer(weights)
        mass = module.canonical_mass(weights, canonical, z)
        total, canon, probs = brute_force(weights, canonical, max_len=10)
        assert abs(z - total) / total < 1e-4, (z, total)
        assert abs(mass - canon / total) < 1e-4, (mass, canon / total)
        strings = [s for s in probs if len(s) > 0]
        flat = np.concatenate([np.asarray(s) for s in strings])
        offsets = np.concatenate(([0], np.cumsum([len(s) for s in strings])))
        prev, nxt, owner = module.cohort_transitions(flat, offsets, k)
        log2p = module.string_log2_probs(weights, z, prev, nxt, owner, len(strings))
        expected = np.log2([probs[s] / z for s in strings])
        assert np.allclose(log2p, expected, atol=1e-9), (log2p[:4], expected[:4])
        bad = weights.copy()
        bad[0, :] = 0.0
        bad[0, 0] = 1.0
        try:
            module.solve_normalizer(bad)
        except module.InvalidSubmission:
            pass
        else:
            raise AssertionError("non-terminating model was accepted")
        print(f"{module.__name__}: Z={z:.6f} brute={total:.6f} canonical={mass:.6f} brute={canon / total:.6f}")

    # Sampler and importance-sampling estimate of Z for a tiny base model.
    base = random_model(rng, k, eos=0.5)
    base /= base.sum(axis=1, keepdims=True)
    local, global_w, diag = conditioning(base, canonical, samples=40000, max_tokens=200, seed=1)
    exact = diag["log2_Z_exact"]
    assert abs(diag["log2_Z_importance_sampling"] - exact) < 0.02, diag
    assert diag["local_sample_canonical_rate"] == 1.0 and diag["global_resample_canonical_rate"] == 1.0, diag
    z_true = test_eval.solve_normalizer(np.where(np.pad(canonical, ((0, 1), (0, 1)), constant_values=True), base, 0.0))
    assert abs(math.log2(z_true) - exact) < 1e-12
    strings, _, _ = sample_strings(local, 60000, np.random.default_rng(2), 200)
    freq = sum(1 for s in strings if s.tolist() == [0]) / len(strings)
    p0 = local[k, 0] * local[0, k]
    assert abs(freq - p0) < 5 * math.sqrt(p0 * (1 - p0) / len(strings)), (freq, p0)
    print(f"sampler ok: IS log2Z={diag['log2_Z_importance_sampling']:.5f} exact={exact:.5f}; P([0]) emp={freq:.4f} exact={p0:.4f}")
    print("selftest passed")


if __name__ == "__main__":
    main()
