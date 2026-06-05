"""Run only the unconstrained token bigram baseline artifact."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/baseline/wikitext2_bigram.json")
    args, rest = parser.parse_known_args()
    tmp = Path("artifacts/baseline/_full_conditioning_artifact.json")
    cmd = ["python", "-m", "method.canonical_conditioning", "--output", str(tmp), *rest]
    subprocess.run(cmd, check=True)
    data = json.loads(tmp.read_text())
    for rec in data["records"]:
        rec.pop("local_log2_prob", None)
        rec.pop("local_log2_weight", None)
    data["experiment"] = "wikitext2_bigram_conditioning"
    data["metadata"]["baseline_only"] = True
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
