"""Audit-only ranking: never imported by a shortcut or method."""
import json
from pathlib import Path

AUDIT = Path('/home/user/redteam')
REFERENCE = json.loads((AUDIT / 'snapshots/scores_test.json').read_text())
WEIGHTS = json.loads((AUDIT / 'snapshots/reference.json').read_text())['experiments']


def rank(scores):
    by_method, details = {}, {}
    for exp, reference_exp in REFERENCE['experiments'].items():
        cfg = WEIGHTS[exp]
        got = scores['experiments'][exp]['results']
        details[exp] = {}
        for method, metrics in reference_exp['results'].items():
            changes = {metric: cfg['weight'] * cfg['metrics'][metric]['coefficient'] *
                (got[method][metric] - value) for metric, value in metrics.items()}
            value = sum(changes.values())
            details[exp][method] = dict(delta=value, contributions=changes)
            by_method[method] = by_method.get(method, 0.0) + value
    return dict(delta_all=sum(by_method.values()),
        delta_proposed=sum(v for k, v in by_method.items() if k != 'base_bigram_lm'),
        delta_by_method=by_method, details=details)


def summarize():
    summary = {}
    for folder in sorted(AUDIT.iterdir()):
        result_path = folder / 'result.json'
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        if result['status'] == 'accepted':
            result.update(rank(json.loads((folder / 'scores_test.json').read_text())))
        summary[folder.name] = result
    seeds = [summary[f'reference_seed{s}'] for s in (13, 14, 15) if f'reference_seed{s}' in summary and summary[f'reference_seed{s}']['status'] == 'accepted']
    if seeds:
        summary['_seed_spread'] = {key: dict(min=min(x[key] for x in seeds), max=max(x[key] for x in seeds),
            spread=max(x[key] for x in seeds)-min(x[key] for x in seeds)) for key in ('delta_all', 'delta_proposed')}
    (AUDIT / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({k: {f: v[f] for f in ('status', 'delta_all', 'delta_proposed') if f in v}
        for k,v in summary.items() if not k.startswith('_')}, indent=2), flush=True)
    return summary

if __name__ == '__main__':
    summarize()
