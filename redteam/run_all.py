"""One CPU job, checkpointing each independent test invocation."""
import json
from pathlib import Path
from redteam.run_case import run
from redteam.summarize import summarize

CASES = [
    'as_found', 'reference_seed13', 'reference_seed14', 'reference_seed15',
    'more_training', 'less_training', 'smoothing_fix', 'independent_baseline',
    'shortcut_claimed_zero_loss', 'shortcut_impossible_probability',
    'shortcut_baseline_inflation', 'shortcut_vocab_metadata',
    'shortcut_token_counts', 'shortcut_empty_samples', 'shortcut_easy_cohort',
    'shortcut_test_memorization', 'shortcut_constraint_tradeoff',
    'shortcut_fixed_tokenizer', 'shortcut_metadata_flags',
    'shortcut_null_probability', 'shortcut_nan_probability',
    'shortcut_empty_sample_list', 'shortcut_score_file',
]

if __name__ == '__main__':
    try:
        for name in CASES:
            status = Path('redteam') / name / 'result.json'
            if status.exists():
                result = json.loads(status.read_text())
                if result.get('status') != 'runner_error':
                    print(f'Checkpoint already present: {name}', flush=True)
                    continue
            print(f'RUNNING {name}', flush=True)
            run(name)
            summarize()
    finally:
        summarize()
