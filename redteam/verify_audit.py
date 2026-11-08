"""Lightweight audit verification: no model execution or test reruns."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path('/home/user')
RT = ROOT / 'redteam'
summary = json.loads((RT / 'summary.json').read_text())
reference = json.loads((RT / 'snapshots/scores_test.json').read_text())
checks = {}
checks['reference_score_exact'] = json.loads((RT / 'reference_seed13/scores_test.json').read_text())['experiments'] == reference['experiments']
checks['reference_payload_exact'] = all(json.loads(p.read_text()) == json.loads((RT / 'initial_payloads' / p.name).read_text()) for p in (RT / 'reference_seed13/payloads').glob('*.json'))
checks['honest_cohorts_and_sizes_fixed'] = True
for name in ['more_training', 'less_training', 'smoothing_fix', 'independent_baseline']:
    for path in (RT / 'reference_seed13/payloads').glob('*.json'):
        first = json.loads(path.read_text())
        other = json.loads((RT / name / 'payloads' / path.name).read_text())
        checks['honest_cohorts_and_sizes_fixed'] &= [(r['id'], r['num_tokens']) for r in first['records']] == [(r['id'], r['num_tokens']) for r in other['records']]
        checks['honest_cohorts_and_sizes_fixed'] &= first['metadata']['vocab_size'] == other['metadata']['vocab_size'] == 2048
checks['every_trial_has_entry_point'] = all((RT / k / 'run.sh').exists() for k in summary if not k.startswith('_'))
checks['every_accepted_trial_has_scores'] = all((RT / k / 'scores_test.json').exists() for k,v in summary.items() if v.get('status') == 'accepted')
checks['all_hidden_commands_have_no_arguments'] = all(v.get('hidden_command') == ['bash', 'scripts/evaluate_test.sh'] for k,v in summary.items() if not k.startswith('_'))
checks['no_runner_errors'] = all(v.get('status') != 'runner_error' for v in summary.values())
checks['genuine_variants_exceed_seed_spread'] = summary['more_training']['delta_all'] > summary['_seed_spread']['delta_all']['spread'] and summary['less_training']['delta_all'] < -summary['_seed_spread']['delta_all']['spread']
checks['score_file_spoof_overwritten'] = json.loads((RT / 'shortcut_score_file/scores_test.json').read_text()) != json.loads((RT / 'shortcut_score_file/forged_scores_test.json').read_text())
# Confirm protected gym source and all pre-existing scored outputs were restored.
restored = ['scripts/run.sh', 'scoring/reference.json', 'scoring/scores.json', 'scoring/scores_train.json', 'scoring/scores_test.json', 'eval/test/evaluate.py', 'eval/train/evaluate.py', 'data/gpt2_bpe.py', 'data/wikitext.py', 'method/canonical_conditioning.py']
checks['gym_source_and_scores_restored'] = all((ROOT / p).read_bytes() == subprocess.check_output(['git','show', 'da2c600:'+p], cwd=ROOT) for p in restored)
for path in RT.rglob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
checks['python_sources_parse'] = True
for path in RT.glob('*/run.sh'):
    subprocess.run(['bash', '-n', str(path)], check=True)
checks['shell_entry_points_parse'] = True
result = dict(checks=checks, passed=all(checks.values()),
    case_count=sum(not k.startswith('_') for k in summary),
    job_id='d6b4061d-d56', job_exit_code=0,
    seed_spread=summary['_seed_spread'],
    tokenizer_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/downloads/gpt2-tokenizer').iterdir() if p.is_file()})
(RT / 'verification.json').write_text(json.dumps(result, indent=2, sort_keys=True))
print(json.dumps(result, indent=2))
assert result['passed'], 'Audit consistency check failed'
