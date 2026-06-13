"""Visible train-slice scorer for canonical-conditioning artifacts.

This evaluator reads method artifacts only. It intentionally avoids depending
on the reproduced implementation so any future method can participate by
writing the same artifact shape used by scripts/evaluate.sh.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from data.gpt2_bpe import load_default_tokenizer

SLICE = "train"
SLICE_PARITY = 0


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


def score_artifact(path: Path) -> tuple[str, dict[str, dict[str, float]]]:
    data = json.loads(path.read_text())
    experiment = data.get("experiment")
    if not experiment:
        raise KeyError(f"{path}: missing required experiment field")
    records = select_slice(data["records"])
    metadata = data["metadata"]
    samples = data.get("samples", {})
    baseline_loss = bits_per_string(records, "baseline_log2_prob")
    baseline_token_loss = bits_per_token(records, "baseline_log2_prob")

    results: dict[str, dict[str, float]] = {
        "base_bigram_lm": {
            "relative_log_loss_to_baseline": 1.0,
            "log_loss_bits_per_string": baseline_loss,
            "loss_improvement_bits_per_string": 0.0,
            "log_loss_bits_per_token": baseline_token_loss,
            "finite_record_rate": finite_record_rate(records, "baseline_log2_prob"),
            "canonical_sample_rate": verified_canonical_sample_rate(samples.get("base_bigram_lm", [])),
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
            "loss_improvement_bits_per_string": baseline_loss - local_loss,
            "log_loss_bits_per_token": bits_per_token(records, "local_log2_prob"),
            "finite_record_rate": finite_record_rate(records, "local_log2_prob"),
            "canonical_sample_rate": verified_canonical_sample_rate(samples.get("local_canonical_bigram_lm", [])),
            "vocab_size": metadata["vocab_size"],
        }
        results["global_canonical_bigram_lm"] = {
            "relative_log_loss_to_baseline": global_loss / baseline_loss,
            "log_loss_bits_per_string": global_loss,
            "loss_improvement_bits_per_string": baseline_loss - global_loss,
            "log_loss_bits_per_token": baseline_token_loss + (len(records) * log2_z / token_count(records)),
            "finite_record_rate": finite_record_rate(records, "baseline_log2_prob"),
            "canonical_sample_rate": verified_canonical_sample_rate(samples.get("global_canonical_bigram_lm", [])),
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
    parser.add_argument("--output", default="scoring/scores_train.json")
    args = parser.parse_args()

    artifacts = discover_artifacts(Path(args.artifact))
    results = merge_results(artifacts)
    payload = {
        "_metadata": {
            "slice": SLICE,
            "split_rule": "record id parity; even ids are visible train",
            "artifacts": [str(path) for path in artifacts],
        },
        "experiments": {name: {"results": exp_results} for name, exp_results in sorted(results.items())},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
