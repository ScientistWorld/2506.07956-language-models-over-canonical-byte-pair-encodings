"""Score every hardening case exactly like the hidden test and summarize the deltas.

For each case the submissions are swapped into ``artifacts/`` and the
unmodified ``bash scripts/evaluate_test.sh`` / ``evaluate_train.sh`` are run
with no arguments. The reference artifacts are restored afterwards and the
final reference scores are regenerated. Results are checkpointed per case.

Usage: python -m redteam.hardened.run_cases [case ...]
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from data.bigram_task import CORPORA, ROOT
from redteam.hardened import cases as case_lib

HARD = ROOT / "redteam" / "hardened"
REF_DIR = HARD / "reference"
ARTIFACTS = ROOT / "artifacts"
SCORING = ROOT / "scoring"
REFERENCE_JSON = json.loads((SCORING / "reference.json").read_text())
PROPOSED = ("local_canonical_bigram_lm", "global_canonical_bigram_lm")
PER_STRING_KINDS = {"variant", "baseline", "shortcut"}
EXTRA_CASES = {"tamper_tokenizer_and_support": "shortcut", "score_file_forgery": "shortcut"}


def log(message: str) -> None:
    print(f"[run-cases] {message}", flush=True)


def sh(cmd: list[str], logfile: Path) -> int:
    with open(logfile, "a") as handle:
        handle.write(f"$ {' '.join(cmd)}\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
        handle.write(f"[exit {proc.returncode}]\n")
    return proc.returncode


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(root: Path | None) -> None:
    if ARTIFACTS.exists():
        shutil.rmtree(ARTIFACTS)
    if root is None:
        ARTIFACTS.mkdir()
    else:
        shutil.copytree(root, ARTIFACTS)


def score_installed(case_dir: Path, per_string: bool) -> dict:
    """Run the hidden-test and train-side scripts with no arguments on the installed artifacts."""
    out = {}
    logfile = case_dir / "eval.log"
    for side, script, produced in (("test", "scripts/evaluate_test.sh", "scores_test.json"), ("train", "scripts/evaluate_train.sh", "scores_train.json")):
        code = sh(["bash", script], logfile)
        out[f"{side}_exit"] = code
        if code == 0:
            shutil.copy(SCORING / produced, case_dir / produced)
    if per_string:
        sh(
            [sys.executable, "-I", "eval/test/evaluate.py", "artifacts", "--output", str(case_dir / "scores_test_direct.json"),
             "--per-string-out", str(case_dir / "per_string_test.json")],
            logfile,
        )
    return out


def composite(scores: dict, reference: dict) -> dict:
    """Sum of coefficient x (case - reference) over reference rows, weighted by experiment."""
    result = {"all": 0.0, "proposed": 0.0, "by_row": {}}
    for exp, spec in REFERENCE_JSON["experiments"].items():
        weight = spec["weight"]
        ref_rows = reference["experiments"][exp]["results"]
        case_rows = scores["experiments"].get(exp, {}).get("results", {})
        for method, ref_row in ref_rows.items():
            row = case_rows.get(method)
            if row is None:
                result["by_row"][f"{exp}/{method}"] = None
                continue
            delta = weight * sum(spec["metrics"][m]["coefficient"] * (row[m] - ref_row[m]) for m in spec["metrics"])
            result["by_row"][f"{exp}/{method}"] = delta
            result["all"] += delta
            if method in PROPOSED:
                result["proposed"] += delta
    return result


def bootstrap(case_ps: dict, ref_ps: dict, case_scores: dict, ref_scores: dict, draws: int = 2000) -> dict:
    """Paired bootstrap over held-out strings of the composite delta (proposed rows)."""
    rng = np.random.default_rng(0)
    deltas = np.zeros(draws)
    paired = {}
    for exp, spec in REFERENCE_JSON["experiments"].items():
        corpus = next(c for c, v in CORPORA.items() if v["experiment"] == exp)
        with np.load(ROOT / "eval/test/assets" / f"{corpus}.npz") as data:
            lengths = np.diff(data["cohort_offsets"]).astype(np.float64)
        n = len(lengths)
        idx = rng.integers(0, n, size=(draws, n))
        base = np.asarray(ref_ps[exp]["base_bigram_lm"])[idx]
        lb = -base.mean(axis=1)
        tokens = lengths[idx].sum(axis=1)
        for method in PROPOSED:
            contrib = {}
            for label, ps, sc in (("case", case_ps, case_scores), ("ref", ref_ps, ref_scores)):
                x = np.asarray(ps[exp][method])[idx]
                loss = -x.mean(axis=1)
                metrics = {
                    "relative_log_loss_to_baseline": loss / lb,
                    "log_loss_bits_per_string": loss,
                    "loss_improvement_bits_per_string": lb - loss,
                    "log_loss_bits_per_token": -x.sum(axis=1) / tokens,
                    "canonical_sample_rate": sc["experiments"][exp]["results"][method]["canonical_sample_rate"],
                }
                contrib[label] = sum(spec["metrics"][m]["coefficient"] * metrics[m] for m in spec["metrics"])
            deltas += spec["weight"] * (contrib["case"] - contrib["ref"])
            diff = np.asarray(case_ps[exp][method]) - np.asarray(ref_ps[exp][method])  # bits saved per string
            paired[f"{exp}/{method}"] = {
                "mean_bits_saved_per_string": float(diff.mean()),
                "paired_se": float(diff.std(ddof=1) / np.sqrt(n)),
                "strings_improved_fraction": float((diff > 0).mean()),
            }
    return {
        "delta_proposed_mean": float(deltas.mean()),
        "delta_proposed_ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "prob_delta_positive": float((deltas > 0).mean()),
        "paired_per_row": paired,
    }


def write_run_sh(case_dir: Path, case: str) -> None:
    rel = case_dir.relative_to(ROOT)
    (case_dir / "run.sh").write_text(
        "#!/bin/bash\n# Rebuild this case's submissions and score them like the hidden test.\nset -e\ncd /home/user\n"
        f"python -m redteam.hardened.cases {case} {rel}/artifacts\n"
        "# swap them into artifacts/ (keep a copy of yours first), then run with no arguments:\n"
        f"#   rm -rf artifacts && cp -r {rel}/artifacts artifacts && bash scripts/evaluate_test.sh\n"
    )


def run_tamper(case_dir: Path) -> dict:
    """Replace the visible tokenizer merges and canonical tables, submit an unconstrained model."""
    targets = [ROOT / "data/downloads/gpt2-tokenizer/merges.txt"] + [ROOT / "data/support" / f"{c}_canonical.npz" for c in CORPORA]
    backup = case_dir / "backup"
    backup.mkdir(parents=True, exist_ok=True)
    before = {str(p): sha(p) for p in targets}
    for p in targets:
        shutil.copy(p, backup / p.name)
    try:
        case_lib.build("unconstrained_kneser_ney", case_dir / "artifacts")
        targets[0].write_text("#version: 0.2\n")
        for c in CORPORA:
            support, canonical = case_lib.task(c)
            np.savez_compressed(ROOT / "data/support" / f"{c}_canonical.npz", canonical_packed=np.packbits(np.ones_like(canonical), axis=1),
                                support_size=np.asarray(len(support)), support=support)
        install(case_dir / "artifacts")
        out = score_installed(case_dir, per_string=False)
    finally:
        for p in targets:
            shutil.copy(backup / p.name, p)
        shutil.rmtree(backup)
    after = {str(p): sha(p) for p in targets}
    out["restored"] = before == after
    return {"kind": "shortcut", "info": {"model": "unconstrained Kneser-Ney; visible tokenizer merges and canonical tables replaced to call every pair canonical"}, **out}


def run_forgery(case_dir: Path) -> dict:
    install(REF_DIR / "artifacts")
    fake = json.loads((REF_DIR / "scores_test.json").read_text())
    for exp in fake["experiments"].values():
        for row in exp["results"].values():
            row["log_loss_bits_per_string"] = 0.0
            row["relative_log_loss_to_baseline"] = 0.0
    (SCORING / "scores_test.json").write_text(json.dumps(fake))
    out = score_installed(case_dir, per_string=False)
    out["forged_file_overwritten"] = json.loads((SCORING / "scores_test.json").read_text()) != fake
    return {"kind": "shortcut", "info": {"model": "reference artifacts; fake perfect scores_test.json written before scoring"}, **out}


def run_case(case: str) -> dict:
    case_dir = HARD / case
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "eval.log").unlink(missing_ok=True)
    start = time.time()
    if case == "tamper_tokenizer_and_support":
        result = run_tamper(case_dir)
    elif case == "score_file_forgery":
        result = run_forgery(case_dir)
    else:
        built = case_lib.build(case, case_dir / "artifacts")
        for diag in (case_dir / "artifacts").glob("*/*.diagnostics.json"):
            shutil.copy(diag, case_dir / f"{diag.parent.name}__{diag.name}")
        install(case_dir / "artifacts")
        result = {**built, **score_installed(case_dir, per_string=built["kind"] in PER_STRING_KINDS)}
    write_run_sh(case_dir, case)
    if (case_dir / "artifacts").exists():
        shutil.rmtree(case_dir / "artifacts")
    result["seconds"] = round(time.time() - start, 1)
    return result


def summarize(results: dict) -> dict:
    ref_test = json.loads((REF_DIR / "scores_test.json").read_text())
    ref_train = json.loads((REF_DIR / "scores_train.json").read_text())
    ref_ps = json.loads((REF_DIR / "per_string_test.json").read_text())
    summary = {"reference": {"test": ref_test["experiments"], "train": ref_train["experiments"]}, "cases": {}}
    for case, result in results.items():
        entry = dict(result)
        case_dir = HARD / case
        for side, ref in (("test", ref_test), ("train", ref_train)):
            path = case_dir / f"scores_{side}.json"
            if path.exists():
                scores = json.loads(path.read_text())
                entry[f"delta_{side}"] = composite(scores, ref)
                entry[f"status_{side}"] = scores["_metadata"]["status"]
                entry[f"rows_{side}"] = {exp: v["results"] for exp, v in scores["experiments"].items()}
        ps_path = case_dir / "per_string_test.json"
        if ps_path.exists() and (case_dir / "scores_test.json").exists():
            entry["bootstrap_test"] = bootstrap(json.loads(ps_path.read_text()), ref_ps, json.loads((case_dir / "scores_test.json").read_text()), ref_test)
        summary["cases"][case] = entry
    (HARD / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    lines = ["| case | kind | Δ test (proposed) | Δ train (proposed) | test status |", "|---|---|---:|---:|---|"]
    for case, entry in summary["cases"].items():
        dt = entry.get("delta_test", {}).get("proposed")
        dr = entry.get("delta_train", {}).get("proposed")
        st = entry.get("status_test", {})
        valid = sorted({f"{m}:{'ok' if s.get('valid') else 'FAILED'}" for e in st.values() for m, s in e.items() if m != "base_bigram_lm"})
        lines.append(f"| {case} | {entry.get('kind')} | {dt if dt is None else f'{dt:+.4f}'} | {dr if dr is None else f'{dr:+.4f}'} | {', '.join(valid)} |")
    (HARD / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    selected = sys.argv[1:] or [*case_lib.CASES, *EXTRA_CASES]
    if not ARTIFACTS.exists():
        raise SystemExit("reference artifacts/ missing: run scripts/method.sh first")
    if not (REF_DIR / "artifacts").exists():
        REF_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ARTIFACTS, REF_DIR / "artifacts")
    results_path = HARD / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    try:
        install(REF_DIR / "artifacts")
        if not (REF_DIR / "per_string_test.json").exists():
            (REF_DIR / "eval.log").unlink(missing_ok=True)
            score_installed(REF_DIR, per_string=True)
        for case in selected:
            if case in results and results[case].get("status") == "ok":
                log(f"checkpoint present: {case}")
                continue
            log(f"RUNNING {case}")
            try:
                result = run_case(case)
                result["status"] = "ok"
            except Exception:
                result = {"status": "runner_error", "error": traceback.format_exc()}
                log(result["error"])
            results[case] = result
            results_path.write_text(json.dumps(results, indent=2, sort_keys=True, default=str) + "\n")
    finally:
        install(REF_DIR / "artifacts")
    summarize(results)
    log("done")


if __name__ == "__main__":
    main()
