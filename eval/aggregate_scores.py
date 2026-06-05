"""Aggregate method artifacts into scoring/scores.json.

The evaluator intentionally reads artifacts only. It does not import method/
or recompute canonicalization, so alternative methods can be evaluated by
writing the same artifact fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def bits_per_string(records: list[dict], field: str) -> float:
    usable = [rec for rec in records if field in rec and rec[field] != float("-inf")]
    return -sum(rec[field] for rec in usable) / len(usable)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="artifacts/canonical_conditioning/wikitext2_bigram.json")
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    data = json.loads(Path(args.artifact).read_text())
    records = data["records"]
    log2_z = data["metadata"].get("log2_z_estimate")
    baseline_loss = bits_per_string(records, "baseline_log2_prob")

    results: dict[str, dict[str, float]] = {
        "base_bigram_lm": {
            "log_loss_bits_per_string": baseline_loss,
            "canonical_sample_rate": data["metadata"].get("base_sample_canonical_rate", 0.0),
            "vocab_size": data["metadata"]["vocab_size"],
        }
    }
    if "local_log2_prob" in records[0]:
        local_loss = bits_per_string(records, "local_log2_prob")
        global_loss = baseline_loss + log2_z
        results["local_canonical_bigram_lm"] = {
            "log_loss_bits_per_string": local_loss,
            "canonical_sample_rate": data["metadata"]["local_sample_canonical_rate"],
            "vocab_size": data["metadata"]["vocab_size"],
        }
        results["global_canonical_bigram_lm"] = {
            "log_loss_bits_per_string": global_loss,
            "canonical_sample_rate": 1.0,
            "vocab_size": data["metadata"]["vocab_size"],
        }

    payload = {"experiments": {"wikitext2_bigram_conditioning": {"results": results}}}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
