"""Independently verify candidate fingerprints and quantify expert clustering."""
import hashlib
import json
from pathlib import Path
import sys

import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT / 'URS'))
from model.Model import Model

params = dict(embedding_dim=128, encoder_layer_num=12, ff_hidden_dim=512,
              logit_clipping=50, demand_max1=True, eval_type='greedy')
names = [name for name, _ in Model(**params).named_parameters()]
bases = {}
for label in ['cpu', '4090', '3090']:
    path = RUN / f'init_{label}_seed3407.pt'
    metadata = json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata['checkpoint_sha256']
    state = torch.load(path, map_location='cpu', weights_only=True)['model_state_dict']
    assert list(state) == names
    for name, value in state.items():
        assert hashlib.sha256(value.numpy().tobytes()).hexdigest() == metadata['parameter_hashes'][name]
    bases[label] = state

changed = {k: bases['4090'][k] != bases['3090'][k] for k in names
           if not torch.equal(bases['4090'][k], bases['3090'][k])}
assert len(changed) == 4 and sum(int(mask.sum()) for mask in changed.values()) == 348160
vector = lambda state: torch.cat([state[k][mask].double() for k, mask in changed.items()])
va, vb = vector(bases['4090']), vector(bases['3090'])
old = json.loads((RUN / 'candidate_comparison.json').read_text())['checkpoints']
summary, vectors = {}, {}
for path in sorted((ROOT / 'URS/pretrained').glob('*_300.pt')):
    with torch.serialization.safe_globals([set]):
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
    state = ckpt['model_state_dict']
    ids = [value for group in ckpt['optimizer_state_dict']['param_groups'] for value in group['params']]
    assert len(ids) == len(names)
    optimizer = ckpt['optimizer_state_dict']['state']
    witness = [name for name, pid in zip(names, ids)
               if pid not in optimizer and state[name].std(unbiased=False).item() > 0]
    counts = {label: sum(state[name].numel() for name in witness
                        if torch.equal(state[name], base[name])) for label, base in bases.items()}
    total = sum(state[name].numel() for name in witness)
    assert counts['4090'] == counts['3090'] == total > 0 and counts['cpu'] == 0
    parameter_ids = dict(zip(names, ids))
    extra = {}
    for key, mask in changed.items():
        moments = optimizer[parameter_ids[key]]
        zero = (moments['exp_avg'] == 0) & (moments['exp_avg_sq'] == 0)
        extra[key] = {
            'zero_moment_diagnostic_coordinates': int((zero & mask).sum()),
            'exact_candidate_a_coordinates': int(((state[key] == bases['4090'][key]) & mask).sum()),
            'exact_candidate_b_coordinates': int(((state[key] == bases['3090'][key]) & mask).sum()),
        }
        assert all(value == 0 for value in extra[key].values())
    trained = vector(state)
    cos = lambda v: float(torch.nn.functional.cosine_similarity(trained, v, dim=0))
    ca, cb = cos(va), cos(vb)
    group = '4090' if ca > cb else '3090'
    assert group == old[path.stem]['candidate_initialization_group']
    assert abs(ca-old[path.stem]['ambiguous_coordinates_cosine_4090']) < 1e-5
    assert abs(cb-old[path.stem]['ambiguous_coordinates_cosine_3090']) < 1e-5
    summary[path.stem] = {'exact_witness_coordinates': counts,
                         'candidate_cosine_4090': ca, 'candidate_cosine_3090': cb, 'inferred_group': group,
                         'additional_exact_witness_search': extra}
    vectors[path.stem] = trained / trained.norm()

order = sorted(vectors, key=lambda key: (summary[key]['inferred_group'] == '3090', key))
matrix = torch.stack([vectors[key] for key in order])
pairwise = matrix @ matrix.T
within, cross = [], []
for i, first in enumerate(order):
    for j in range(i):
        other = order[j]
        target = within if summary[first]['inferred_group'] == summary[other]['inferred_group'] else cross
        target.append(float(pairwise[i, j]))
report = {'candidate_hashes_and_all_tensor_hashes_match': True,
          'difference_coordinates': 348160, 'shared_coordinates': 4960487-348160,
          'groups_are_inferred_not_historical_full_base_proof': True,
          'experts': summary, 'expert_order': order, 'pairwise_cosine': pairwise.tolist(),
          'within_group_range': [min(within), max(within)],
          'cross_group_range': [min(cross), max(cross)]}
(RUN / 'evidence_verification.json').write_text(json.dumps(report, indent=2) + '\n')
print('Verified original checkpoints, 3 candidate hashes, all candidate tensor hashes and all untouched random witnesses.')
print('Within-group cosine range:', report['within_group_range'])
print('Cross-group cosine range:', report['cross_group_range'])
print('Full historical initialization remains unproven; four differing tensors retain no exact untouched coordinates.')
