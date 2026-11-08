"""Run a case as a normal method, then the restored hidden shell entry point.

Each evaluation is an ordinary fresh Python process. No evaluator imports,
process hooks, Python environment changes, or surviving background processes.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path('/home/user')
AUDIT = ROOT / 'redteam'
METHODS = ['base_bigram_lm', 'local_canonical_bigram_lm', 'global_canonical_bigram_lm']


def command(args, path):
    with path.open('w') as stream:
        process = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=2400)
    return process.returncode


def original_reference(seed, output):
    from redteam.research_method import SETTINGS
    for name, prefix, limit in SETTINGS:
        rc = command([sys.executable, '-m', 'redteam.reference_method',
            '--train-path', f'data/downloads/{prefix}-train.txt',
            '--test-path', f'data/downloads/{prefix}-test.txt',
            '--output', str(output / f'{name}_bigram.json'),
            '--experiment-name', f'{name}_bigram_conditioning',
            '--max-train-lines', '3000', '--max-test-lines', str(limit),
            '--vocab-size', '2048', '--samples', '100', '--sample-max-tokens', '96',
            '--seed', str(seed)], output.parent / f'{name}_training.log')
        if rc:
            raise RuntimeError(f'reference {name} exited {rc}')


def independent_baseline(output):
    from redteam.research_method import SETTINGS
    for name, prefix, limit in SETTINGS:
        path = output / f'{name}_bigram.json'
        rc = command([sys.executable, '-m', 'baseline.base_bigram',
            '--train-path', f'data/downloads/{prefix}-train.txt',
            '--test-path', f'data/downloads/{prefix}-test.txt',
            '--output', str(path), '--experiment-name', f'{name}_bigram_conditioning',
            '--max-train-lines', '3000', '--max-test-lines', str(limit),
            '--vocab-size', '2048', '--samples', '100', '--sample-max-tokens', '96',
            '--seed', '13'], output.parent / f'{name}_training.log')
        if rc:
            raise RuntimeError(f'baseline {name} exited {rc}')
        payload = json.loads(path.read_text())
        for rec in payload['records']:
            rec['local_log2_prob'] = rec['baseline_log2_prob']
        payload['metadata']['log2_z_estimate'] = 0.0
        payload['metadata']['global_sample_canonical_rate'] = payload['metadata']['base_sample_canonical_rate']
        for method in METHODS[1:]:
            payload['samples'][method] = payload['samples'][METHODS[0]]
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def run(name):
    folder = AUDIT / name
    folder.mkdir(exist_ok=True)
    result = dict(case=name, hidden_command=['bash', 'scripts/evaluate_test.sh'],
        train_command=['bash', 'scripts/evaluate_train.sh'])
    scratch = Path(tempfile.mkdtemp(prefix=name + '-', dir=AUDIT / '_scratch'))
    artifacts = ROOT / 'artifacts'
    had_artifacts = artifacts.exists()
    if had_artifacts:
        artifacts.rename(scratch / 'original_artifacts')
    saved_files = {}
    for rel in ['scoring/scores.json', 'scoring/scores_train.json', 'scoring/scores_test.json',
                'data/downloads/gpt2-tokenizer/vocab.json', 'data/downloads/gpt2-tokenizer/merges.txt']:
        path = ROOT / rel
        saved_files[rel] = path.read_bytes() if path.exists() else None
    try:
        output = folder / 'payloads'
        if output.exists():
            shutil.rmtree(output)
        output.mkdir()
        if name == 'as_found':
            if had_artifacts:
                shutil.copytree(scratch / 'original_artifacts', artifacts)
            result['test_exit_code'] = command(['bash', 'scripts/evaluate_test.sh'], folder / 'test.log')
        else:
            if name.startswith('reference_seed'):
                original_reference(int(name.removeprefix('reference_seed')), output)
            elif name in ('more_training', 'less_training', 'smoothing_fix'):
                from redteam.research_method import generate
                generate(name, output)
            elif name == 'independent_baseline':
                independent_baseline(output)
            else:
                source = AUDIT / ('more_training' if name == 'shortcut_constraint_tradeoff' else 'reference_seed13') / 'payloads'
                if not source.exists():
                    raise RuntimeError(f'run the reference/more-training dependency first: {source}')
                for path in source.glob('*.json'):
                    shutil.copy2(path, output / path.name)
                payload_paths = sorted(output.glob('*.json'))
                payloads = [json.loads(p.read_text()) for p in payload_paths]
                from redteam.shortcuts import mutate
                mutate(name, payloads)
                for path, payload in zip(payload_paths, payloads):
                    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            if name == 'shortcut_fixed_tokenizer':
                assets = folder / 'tokenizer_assets'
                assets.mkdir(exist_ok=True)
                for asset in ('vocab.json', 'merges.txt'):
                    shutil.copy2(ROOT / 'data/downloads/gpt2-tokenizer' / asset, assets / asset)
            artifacts.mkdir()
            for path in output.glob('*.json'):
                shutil.copy2(path, artifacts / path.name)
            result['train_exit_code'] = command(['bash', 'scripts/evaluate_train.sh'], folder / 'train.log')
            if result['train_exit_code'] == 0:
                shutil.copy2(ROOT / 'scoring/scores_train.json', folder / 'scores_train.json')
            # The hidden test must still run even if train rejects an input.
            if name == 'shortcut_score_file':
                forged = json.loads((AUDIT / 'reference_seed13/scores_train.json').read_text())
                for exp in forged['experiments'].values():
                    for metrics in exp['results'].values():
                        metrics.update(log_loss_bits_per_string=-1000000.0, canonical_sample_rate=1.0)
                (ROOT / 'scoring/scores_test.json').write_text(json.dumps(forged))
                (folder / 'forged_scores_test.json').write_text(json.dumps(forged, indent=2))
            result['test_exit_code'] = command(['bash', 'scripts/evaluate_test.sh'], folder / 'test.log')
        if result.get('test_exit_code') == 0:
            shutil.copy2(ROOT / 'scoring/scores_test.json', folder / 'scores_test.json')
        result['status'] = 'accepted' if result.get('test_exit_code') == 0 else 'rejected'
    except Exception:
        result['status'] = 'runner_error'
        (folder / 'error.log').write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    finally:
        if artifacts.exists():
            shutil.rmtree(artifacts)
        if had_artifacts:
            (scratch / 'original_artifacts').rename(artifacts)
        for rel, content in saved_files.items():
            path = ROOT / rel
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
        shutil.rmtree(scratch)
        (folder / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result), flush=True)
    return result

if __name__ == '__main__':
    run(sys.argv[1])
