"""Reuse verified LOT statistics; apply official CSP to the unscaled LOT delta."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
LOT = RUN.parent / '20261117_lot_samples_seen5'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'URS'))
from merge_lab.ties import load_state
from merge_lab.lot import lot_vector
from merge_lab.csp import base_svd, project_vector
from model.Model import Model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--ids', help='Optional comma-separated model IDs')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    manifest = json.loads((RUN / 'manifest.json').read_text())
    jobs = [x for x in json.loads((RUN / 'evaluation_plan.json').read_text())
            if x['family'] != 'base' and x['seed'] == args.seed and 'source_result' not in x]
    if args.ids:
        jobs = [x for x in jobs if x['id'] in args.ids.split(',')]
        assert len(jobs) == len(args.ids.split(','))
    base = load_state(manifest['base']['path'])
    experts = [load_state(x['checkpoint']) for x in manifest['experts']]
    deltas = [{name: state[name] - value for name, value in base.items()} for state in experts]
    params = json.loads((RUN.parent / '20261116_urs_finetune_seen5/models/cvrp/'
                        'finetune_metadata.json').read_text())['model_params']
    model = Model(**params)
    svd_path = RUN / 'cache' / 'base_svd.pt'
    svd_reused = svd_path.exists()
    svd_seconds = 0.
    if svd_reused:
        svd = torch.load(svd_path, map_location='cpu', weights_only=True)
    else:
        started = time.monotonic()
        svd = base_svd(base)
        torch.cuda.synchronize()
        svd_seconds = time.monotonic() - started
        torch.save(svd, svd_path)
    lot_cache = {}
    for job in jobs:
        record_path = RUN / 'build_records' / f"{job['id']}.json"
        if record_path.exists():
            continue
        size = job['samples']
        feature_path = LOT / 'features' / f'n{size}_s{args.seed}_scale1.0.pt'
        lot_path = RUN / 'cache' / f'lot_n{size}_s{args.seed}.pt'
        lot_reused = size in lot_cache or lot_path.exists()
        lot_seconds = 0.
        if size not in lot_cache:
            if lot_path.exists():
                cached = torch.load(lot_path, map_location='cpu', weights_only=True)
            else:
                feats = torch.load(feature_path, map_location='cpu', weights_only=True)
                assert len({tuple(x) for x in feats['inactive']}) == 1
                assert all(len(x) == size for x in feats['sample_indices'].values())
                started = time.monotonic()
                vector, lot_diagnostics = lot_vector(deltas, feats)
                torch.cuda.synchronize()
                lot_seconds = time.monotonic() - started
                cached = {'vector': vector, 'diagnostics': lot_diagnostics,
                          'sample_indices': feats['sample_indices']}
                torch.save(cached, lot_path)
            lot_cache[size] = cached
        cached = lot_cache[size]
        started = time.monotonic()
        vector, diagnostics = project_vector(cached['vector'], svd, job['k'])
        torch.cuda.synchronize()
        projection_seconds = time.monotonic() - started
        started = time.monotonic()
        state = {name: value + job['scale'] * vector[name] for name, value in base.items()}
        assert all(torch.isfinite(x).all() for x in state.values())
        model.load_state_dict(state, strict=True)
        info = {**job, 'base_checkpoint': manifest['base']['path'],
                'expert_checkpoints': [x['checkpoint'] for x in manifest['experts']],
                'upstream_commit': manifest['upstream_commit'],
                'sample_indices': cached['sample_indices']}
        torch.save({'model_state_dict': state, 'merge_info': info}, job['checkpoint'])
        timing = {'svd_seconds': svd_seconds, 'svd_reused': svd_reused,
                  'lot_seconds': lot_seconds, 'lot_reused': lot_reused,
                  'projection_seconds': projection_seconds, 'projection_reused': False,
                  'save_seconds': time.monotonic() - started}
        record = {**info, 'diagnostics': diagnostics, 'lot_diagnostics': cached['diagnostics'],
                  'timing': timing, 'feature_path': str(feature_path),
                  'lot_cache_path': str(lot_path), 'svd_cache_path': str(svd_path),
                  'gpu': torch.cuda.get_device_name(), 'finite': True, 'strict_load': True,
                  'finished': datetime.now().astimezone().isoformat()}
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
        svd_reused, svd_seconds = True, 0.
        print(f"Built {job['id']}: CSP={projection_seconds:.2f}s, "
              f"retained norm={diagnostics['retained_delta_norm_ratio']:.4f}", flush=True)


if __name__ == '__main__':
    main()
