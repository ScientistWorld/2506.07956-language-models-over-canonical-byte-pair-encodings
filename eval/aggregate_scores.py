"""Aggregate method artifacts into scoring/scores.json.

The evaluator intentionally reads artifacts only. It does not import method/
or recompute canonicalization, so alternative methods can be evaluated by
writing the same artifact fields.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from data.gpt2_bpe import load_default_tokenizer


def bits_per_string(records: list[dict], field: str) -> float:
    values = []
    for rec in records:
        if field not in rec:
            raise KeyError(f"missing required field {field!r} in record {rec.get('id', '<unknown>')}")
        value = rec[field]
        if not math.isfinite(value):
            raise ValueError(f"non-finite {field!r} in record {rec.get('id', '<unknown>')}: {value}")
        values.append(value)
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
        raise ValueError("no tokens available for scoring")
    return -total_log2_prob / total_tokens


def token_count(records: list[dict]) -> int:
    total = 0
    for rec in records:
        num_tokens = int(rec.get("num_tokens", 0))
        if num_tokens <= 0:
            raise ValueError(f"record {rec.get('id', '<unknown>')} has invalid num_tokens={num_tokens}")
        total += num_tokens
    if total <= 0:
        raise ValueError("no tokens available for scoring")
    return total


def finite_record_rate(records: list[dict], field: str) -> float:
    if not records:
        raise ValueError("no records available for scoring")
    return sum(math.isfinite(rec.get(field, float("nan"))) for rec in records) / len(records)


def verified_canonical_sample_rate(samples: list[list[int]]) -> float:
    if not samples:
        raise ValueError("artifact did not provide generated samples for canonicality verification")
    tokenizer = load_default_tokenizer()
    canonical = 0
    for sample in samples:
        canonical += int(
            all(tokenizer.tokenizer_canonical_bigram(left, right) for left, right in zip(sample, sample[1:]))
        )
    return canonical / len(samples)


def score_artifact(path: Path) -> tuple[str, dict[str, dict[str, float]]]:
    data = json.loads(path.read_text())
    records = data["records"]
    if not records:
        raise ValueError(f"{path}: artifact has no scored records")
    experiment = data.get("experiment")
    if not experiment:
        raise KeyError(f"{path}: missing required experiment field")
    samples = data.get("samples", {})
    log2_z = data["metadata"].get("log2_z_estimate")
    method_names = data["metadata"].get(
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
            "vocab_size": data["metadata"]["vocab_size"],
        }
    }
    if "local_log2_prob" in records[0]:
        if log2_z is None:
            raise KeyError(f"{path}: missing log2_z_estimate in artifact metadata")
        local_loss = bits_per_string(records, "local_log2_prob")
        global_loss = baseline_loss + log2_z
        results[local_name] = {
            "relative_log_loss_to_baseline": local_loss / baseline_loss,
            "log_loss_bits_per_string": local_loss,
            "loss_improvement_bits_per_string": baseline_loss - local_loss,
            "log_loss_bits_per_token": bits_per_token(records, "local_log2_prob"),
            "finite_record_rate": finite_record_rate(records, "local_log2_prob"),
            "canonical_sample_rate": verified_canonical_sample_rate(samples.get(local_name, [])),
            "vocab_size": data["metadata"]["vocab_size"],
        }
        results[global_name] = {
            "relative_log_loss_to_baseline": global_loss / baseline_loss,
            "log_loss_bits_per_string": global_loss,
            "loss_improvement_bits_per_string": baseline_loss - global_loss,
            "log_loss_bits_per_token": baseline_token_loss + (len(records) * log2_z / token_count(records)),
            "finite_record_rate": finite_record_rate(records, "baseline_log2_prob"),
            "canonical_sample_rate": verified_canonical_sample_rate(samples.get(global_name, [])),
            "vocab_size": data["metadata"]["vocab_size"],
        }

    return experiment, results


def discover_artifacts(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.exists():
        raise FileNotFoundError(f"artifact path does not exist: {target}")
    artifacts = sorted(target.glob("**/*.json"))
    if not artifacts:
        raise FileNotFoundError(f"no JSON artifacts found under {target}")
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="artifacts/canonical_conditioning/wikitext2_bigram.json")
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    experiments: dict[str, dict[str, dict[str, float]]] = {}
    for artifact in discover_artifacts(Path(args.artifact)):
        experiment, results = score_artifact(artifact)
        if experiment in experiments:
            overlap = set(experiments[experiment]).intersection(results)
            if overlap:
                raise ValueError(f"duplicate methods for {experiment} in artifacts: {sorted(overlap)}")
            experiments[experiment].update(results)
        else:
            experiments[experiment] = results

    payload = {"experiments": {name: {"results": results} for name, results in sorted(experiments.items())}}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
