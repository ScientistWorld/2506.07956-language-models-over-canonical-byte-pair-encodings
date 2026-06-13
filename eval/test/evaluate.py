"""Held-out test-slice scorer for canonical-conditioning artifacts.

This evaluator mirrors eval/train/evaluate.py while keeping an independent
implementation and selecting the complementary held-out record slice.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from data.gpt2_bpe import load_default_tokenizer

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


def bits_per_token(records: list[dict], field: str) -> float:
    total_log2_prob = 0.0
    total_tokens = 0
    for rec in records:
        if field not in rec:
            raise KeyError(f"missing required field {field!r} in record {rec.get('id', '<unknown>')}")
        value = rec[field]
        if not math.isfinite(value):
            raise ValueError(f"non-finite {field!r} in record {rec.get('id', '<unknown>')}: {value}")
        num_tokens = int(rec.get("num_tokens", 0))
        if num_tokens <= 0:
            raise ValueError(f"record {rec.get('id', '<unknown>')} has invalid num_tokens={num_tokens}")
        total_log2_prob += value
        total_tokens += num_tokens
    if total_tokens <= 0:
        raise ValueError(f"no tokens available for {SLICE} slice")
    return -total_log2_prob / total_tokens


def token_count(records: list[dict]) -> int:
    total = 0
    for rec in records:
        num_tokens = int(rec.get("num_tokens", 0))
        if num_tokens <= 0:
            raise ValueError(f"record {rec.get('id', '<unknown>')} has invalid num_tokens={num_tokens}")
        total += num_tokens
    if total <= 0:
        raise ValueError(f"no tokens available for {SLICE} slice")
    return total


def finite_record_rate(records: list[dict], field: str) -> float:
    if not records:
        raise ValueError(f"no records available for {SLICE} slice")
    finite = 0
    for rec in records:
        if field not in rec:
            raise KeyError(f"missing required field {field!r} in record {rec.get('id', '<unknown>')}")
        finite += int(math.isfinite(rec[field]))
    return finite / len(records)


def verified_canonical_sample_rate(samples: list[list[int]]) -> float:
    if not samples:
        raise ValueError(f"artifact did not provide generated samples for {SLICE} canonicality verification")
    tokenizer = load_default_tokenizer()
    canonical = 0
    for sample in samples:
        canonical += int(
            all(tokenizer.tokenizer_canonical_bigram(left, right) for left, right in zip(sample, sample[1:]))
        )
    return canonical / len(samples)


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
    artifacts = sorted(target.glob("**/*.json"))
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


def expected_methods(train_scores_path: Path) -> dict[str, set[str]]:
    if not train_scores_path.exists():
        return {}
    data = json.loads(train_scores_path.read_text())
    experiments = data.get("experiments", data)
    expected: dict[str, set[str]] = {}
    for experiment, exp_data in experiments.items():
        if experiment.startswith("_"):
            continue
        if not isinstance(exp_data, dict):
            raise TypeError(f"{train_scores_path}: experiment {experiment!r} must be a dict")
        methods = exp_data.get("results", exp_data)
        if not isinstance(methods, dict):
            raise TypeError(f"{train_scores_path}: experiment results must be a dict")
        expected[experiment] = {name for name in methods if not name.startswith("_")}
    return expected


def score_artifact(path: Path) -> tuple[str, dict[str, dict[str, float]]]:
    data = json.loads(path.read_text())
    experiment = data.get("experiment")
    if not experiment:
        raise KeyError(f"{path}: missing required experiment field")
    records = select_slice(data["records"])
    metadata = data["metadata"]
    samples = data.get("samples", {})
    method_names = metadata.get(
        "method_names",
        {
            "base": "base_bigram_lm",
            "local": "local_canonical_bigram_lm",
            "global": "global_canonical_bigram_lm",
        },
    )
    base_name = method_names["base"]
    local_name = method_names.get("local", "local_canonical_bigram_lm")
    global_name = method_names.get("global", "global_canonical_bigram_lm")
    baseline_loss = bits_per_string(records, "baseline_log2_prob")
    baseline_token_loss = bits_per_token(records, "baseline_log2_prob")

    results: dict[str, dict[str, float]] = {
        base_name: {
            "relative_log_loss_to_baseline": 1.0,
            "log_loss_bits_per_string": baseline_loss,
            "loss_improvement_bits_per_string": 0.0,
            "log_loss_bits_per_token": baseline_token_loss,
            "finite_record_rate": finite_record_rate(records, "baseline_log2_prob"),
            "canonical_sample_rate": verified_canonical_sample_rate(samples.get(base_name, [])),
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
        results[local_name] = {
            "relative_log_loss_to_baseline": local_loss / baseline_loss,
            "log_loss_bits_per_string": local_loss,
            "loss_improvement_bits_per_string": baseline_loss - local_loss,
            "log_loss_bits_per_token": bits_per_token(records, "local_log2_prob"),
            "finite_record_rate": finite_record_rate(records, "local_log2_prob"),
            "canonical_sample_rate": verified_canonical_sample_rate(samples.get(local_name, [])),
            "vocab_size": metadata["vocab_size"],
        }
        results[global_name] = {
            "relative_log_loss_to_baseline": global_loss / baseline_loss,
            "log_loss_bits_per_string": global_loss,
            "loss_improvement_bits_per_string": baseline_loss - global_loss,
            "log_loss_bits_per_token": baseline_token_loss + (len(records) * log2_z / token_count(records)),
            "finite_record_rate": finite_record_rate(records, "baseline_log2_prob"),
            "canonical_sample_rate": verified_canonical_sample_rate(samples.get(global_name, [])),
            "vocab_size": metadata["vocab_size"],
        }
    return experiment, results


def merge_results(artifacts: list[Path]) -> dict[str, dict[str, dict[str, float]]]:
    merged: dict[str, dict[str, dict[str, float]]] = {}
    for artifact in artifacts:
        experiment, results = score_artifact(artifact)
        exp_results = merged.setdefault(experiment, {})
        for method, scores in results.items():
            if method in exp_results and exp_results[method] != scores:
                raise ValueError(
                    f"conflicting scores for {experiment}.{method}; pass one artifact or use unique method outputs"
                )
            exp_results[method] = scores
    if not merged:
        raise ValueError("no experiments were scored")
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
        missing_experiments = sorted(set(expected) - set(results))
        if missing_experiments:
            raise KeyError(f"test slice is missing experiments scored on train: {missing_experiments}")
        filtered: dict[str, dict[str, dict[str, float]]] = {}
        ignored_extra: dict[str, list[str]] = {}
        for experiment, expected_for_exp in sorted(expected.items()):
            exp_results = results[experiment]
            missing = sorted(expected_for_exp - exp_results.keys())
            extra = sorted(exp_results.keys() - expected_for_exp)
            if missing:
                raise KeyError(f"test slice is missing methods scored on train for {experiment}: {missing}")
            filtered[experiment] = {name: exp_results[name] for name in sorted(expected_for_exp)}
            if extra:
                ignored_extra[experiment] = extra
        results = filtered
    else:
        ignored_extra = {}

    payload = {
        "_metadata": {
            "slice": SLICE,
            "split_rule": "record id parity; odd ids are held-out test",
            "artifacts": [str(path) for path in artifacts],
            "matched_train_scores": bool(expected),
            "ignored_extra_methods": ignored_extra,
        },
        "experiments": {name: {"results": exp_results} for name, exp_results in sorted(results.items())},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
