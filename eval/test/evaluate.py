"""Held-out test-slice scorer for WikiText-2 bigram conditioning artifacts.

This evaluator mirrors eval/train/evaluate.py while keeping an independent
implementation and selecting the complementary held-out record slice.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EXPERIMENT = "wikitext2_bigram_conditioning"
SLICE = "test"
SLICE_PARITY = 1


def bits_per_string(records: list[dict], field: str) -> float:
    values = []
    for rec in records:
        if field not in rec:
            raise KeyError(f"missing required field {field!r} in record {rec.get('id', '<unknown>')}")
        value = rec[field]
        if not math.isfinite(value):
            raise ValueError(f"non-finite {field!r} in record {rec.get('id', '<unknown>')}: {value}")
        values.append(value)
    if not values:
        raise ValueError(f"no records available for {SLICE} slice")
    return -sum(values) / len(values)


def record_id(rec: dict) -> int:
    if "id" not in rec:
        raise KeyError("artifact record is missing required 'id' field")
    return int(rec["id"])


def select_slice(records: list[dict]) -> list[dict]:
    return [rec for rec in records if record_id(rec) % 2 == SLICE_PARITY]


def discover_artifacts(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.exists():
        raise FileNotFoundError(f"artifact path does not exist: {target}")
    artifacts = sorted(target.glob("*/wikitext2_bigram.json")) + sorted(target.glob("*.json"))
    unique = []
    seen = set()
    for artifact in artifacts:
        resolved = artifact.resolve()
        if resolved not in seen:
            unique.append(artifact)
            seen.add(resolved)
    if not unique:
        raise FileNotFoundError(f"no JSON artifacts found under {target}")
    return unique


def expected_methods(train_scores_path: Path) -> set[str]:
    if not train_scores_path.exists():
        return set()
    data = json.loads(train_scores_path.read_text())
    experiments = data.get("experiments", data)
    exp = experiments.get(EXPERIMENT)
    if not isinstance(exp, dict):
        raise KeyError(f"{train_scores_path}: missing experiment {EXPERIMENT!r}")
    methods = exp.get("results", exp)
    if not isinstance(methods, dict):
        raise TypeError(f"{train_scores_path}: experiment results must be a dict")
    return {name for name in methods if not name.startswith("_")}


def score_artifact(path: Path) -> dict[str, dict[str, float]]:
    data = json.loads(path.read_text())
    records = select_slice(data["records"])
    metadata = data["metadata"]
    baseline_loss = bits_per_string(records, "baseline_log2_prob")

    results: dict[str, dict[str, float]] = {
        "base_bigram_lm": {
            "relative_log_loss_to_baseline": 1.0,
            "log_loss_bits_per_string": baseline_loss,
            "canonical_sample_rate": metadata.get("base_sample_canonical_rate", 0.0),
            "vocab_size": metadata["vocab_size"],
        }
    }

    if records and "local_log2_prob" in records[0]:
        log2_z = metadata.get("log2_z_estimate")
        if log2_z is None:
            raise KeyError(f"{path}: missing log2_z_estimate in method artifact metadata")
        local_loss = bits_per_string(records, "local_log2_prob")
        global_loss = baseline_loss + log2_z
        global_canonical_rate = metadata.get(
            "global_sample_canonical_rate", metadata.get("local_sample_canonical_rate")
        )
        if global_canonical_rate is None:
            raise KeyError(f"{path}: missing global_sample_canonical_rate in method artifact metadata")
        results["local_canonical_bigram_lm"] = {
            "relative_log_loss_to_baseline": local_loss / baseline_loss,
            "log_loss_bits_per_string": local_loss,
            "canonical_sample_rate": metadata["local_sample_canonical_rate"],
            "vocab_size": metadata["vocab_size"],
        }
        results["global_canonical_bigram_lm"] = {
            "relative_log_loss_to_baseline": global_loss / baseline_loss,
            "log_loss_bits_per_string": global_loss,
            "canonical_sample_rate": global_canonical_rate,
            "vocab_size": metadata["vocab_size"],
        }
    return results


def merge_results(artifacts: list[Path]) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    for artifact in artifacts:
        for method, scores in score_artifact(artifact).items():
            if method in merged and merged[method] != scores:
                raise ValueError(
                    f"conflicting scores for method {method!r}; pass one artifact or use unique method outputs"
                )
            merged[method] = scores
    if not merged:
        raise ValueError("no methods were scored")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", nargs="?", default="artifacts")
    parser.add_argument("--output", default="scoring/scores_test.json")
    parser.add_argument("--train-scores", default="scoring/scores_train.json")
    args = parser.parse_args()

    artifacts = discover_artifacts(Path(args.artifact))
    results = merge_results(artifacts)
    expected = expected_methods(Path(args.train_scores))
    if expected:
        missing = sorted(expected - results.keys())
        extra = sorted(results.keys() - expected)
        if missing:
            raise KeyError(f"test slice is missing methods scored on train: {missing}")
        results = {name: results[name] for name in sorted(expected)}
    else:
        extra = []

    payload = {
        "_metadata": {
            "slice": SLICE,
            "split_rule": "record id parity; odd ids are held-out test",
            "artifacts": [str(path) for path in artifacts],
            "matched_train_scores": bool(expected),
            "ignored_extra_methods": extra,
        },
        "experiments": {EXPERIMENT: {"results": results}},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
