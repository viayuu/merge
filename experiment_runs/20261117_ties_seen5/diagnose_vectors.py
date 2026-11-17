"""Describe task-vector alignment and the official trimming thresholds."""

import json
from pathlib import Path
import sys
import torch

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parents[1]))
from merge_lab.ties import prepare_vectors
from third_party.ties_merging.minimal import (
    topk_values_mask, resolve_sign, disjoint_merge)

torch.set_num_threads(4)
manifest = json.loads((RUN / 'manifest.json').read_text())
_, base, deltas = prepare_vectors(manifest['base']['path'],
                                 [e['checkpoint'] for e in manifest['experts']])
# FP64 is only for these descriptive norms/cosines; merging remains official FP32.
normalized = torch.nn.functional.normalize(deltas.double(), dim=1)
active = (deltas != 0).any(dim=0)
conflict = (deltas > 0).any(dim=0) & (deltas < 0).any(dim=0)
report = {
    'tasks': manifest['task_order'], 'base_l2_norm': base.double().norm().item(),
    'task_delta_l2_norm': deltas.double().norm(dim=1).tolist(),
    'average_delta_l2_norm': deltas.mean(dim=0).double().norm().item(),
    'task_delta_cosine': (normalized @ normalized.T).tolist(),
    'active_coordinate_fraction': active.double().mean().item(),
    'raw_conflicting_fraction_of_active_coordinates': (conflict.sum() / active.sum()).item(),
    'by_keep_percent': [],
}
for keep in [10, 20, 30, 50, 70, 90]:
    trimmed, kept = topk_values_mask(deltas, K=keep / 100)
    active = (trimmed != 0).any(dim=0)
    conflict = (trimmed > 0).any(dim=0) & (trimmed < 0).any(dim=0)
    update = disjoint_merge(trimmed, 'dis-mean', resolve_sign(trimmed))
    report['by_keep_percent'].append({
        'keep_percent': keep,
        'actual_threshold_mask_fraction_by_expert': kept.tolist(),
        'active_coordinate_fraction': active.double().mean().item(),
        'conflicting_fraction_of_active_coordinates': (conflict.sum() / active.sum()).item(),
        'ties_update_l2_norm_scale1': update.double().norm().item(),
    })
(RUN / 'merge_diagnostics.json').write_text(json.dumps(report, indent=2) + '\n')
print('Saved merge_diagnostics.json')
