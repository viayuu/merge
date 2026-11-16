"""Evaluate checkpoints through the original URS test.py and summarize paper gaps."""

import argparse
from pathlib import Path
import subprocess
import sys

from summarize import summarize


ROOT = Path(__file__).resolve().parents[1]
URS = ROOT / 'URS'


def main():
    parser = argparse.ArgumentParser(
        description='Run original URS evaluation. Extra options are passed to URS/test.py.',
        epilog='Example: --checkpoints model.pt --out-dir results/run1 --problem_set cvrp,cvrptw',
    )
    parser.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--data_dir', type=Path, default=URS / 'dataset')
    args, urs_args = parser.parse_known_args()
    if '--model_load' in urs_args or '--output_json' in urs_args:
        parser.error('Use --checkpoints and --out-dir instead of --model_load/--output_json')

    # Resolve relative paths before switching to the original repository's cwd.
    checkpoints = [path.resolve() for path in args.checkpoints]
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if len({p.stem for p in checkpoints}) != len(checkpoints):
        parser.error('Checkpoint filenames must be distinct within one evaluation run')
    results = []
    for checkpoint in checkpoints:
        output = out_dir / (checkpoint.stem + '.json')
        command = [
            sys.executable, str(URS / 'test.py'),
            '--data_dir', str(args.data_dir.resolve()),
            '--problem_set', 'train_problem_list',
            '--test_scale_list', '100', '--test_episodes', '1000',
            *urs_args,
            '--model_load', str(checkpoint), '--output_json', str(output),
        ]
        print(f'Evaluating {checkpoint.name}', flush=True)
        subprocess.run(command, cwd=URS, check=True)
        results.append(output)
        summarize(results, out_dir)
    print(f'Results: {out_dir / "comparison.md"}')


if __name__ == '__main__':
    main()
