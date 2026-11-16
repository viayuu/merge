"""Read checkpoint optimizer states to locate unchanged initialization fingerprints."""
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'URS'))
from model.Model import Model

model = Model(embedding_dim=128, encoder_layer_num=12, ff_hidden_dim=512,
              logit_clipping=50, demand_max1=True, eval_type='greedy')
names = [name for name, _ in model.named_parameters()]
report = {}
for path in sorted((ROOT/'URS/pretrained').glob('*.pt')):
    with torch.serialization.safe_globals([set]):
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
    groups = ckpt['optimizer_state_dict']['param_groups']
    param_ids = [i for group in groups for i in group['params']]
    assert len(param_ids) == len(names)
    optimizer = ckpt['optimizer_state_dict']['state']
    unused = [name for name, key in zip(names, param_ids) if key not in optimizer]
    fingerprints = {name: hashlib.sha256(ckpt['model_state_dict'][name].contiguous().numpy().tobytes()).hexdigest()
                    for name in unused if ckpt['model_state_dict'][name].std(unbiased=False).item() > 0}
    steps = sorted({float(value['step']) for value in optimizer.values() if 'step' in value})
    item = {
        'checkpoint_keys': list(ckpt), 'epoch': ckpt['epoch'],
        'optimizer_param_count': len(param_ids), 'state_count': len(optimizer),
        'optimizer_steps': steps, 'unused_parameters': unused,
        'unused_random_fingerprints': fingerprints,
        'unused_random_numel': sum(ckpt['model_state_dict'][n].numel() for n in fingerprints),
        'optimizer_settings': [{k: v for k,v in group.items() if k != 'params'} for group in groups],
    }
    report[path.stem] = item
    print(path.stem, 'epoch',item['epoch'],'states',len(optimizer),'unused random tensors',len(fingerprints),
          'elements',item['unused_random_numel'],'steps',steps,flush=True)
(ROOT/'experiment_runs/20261116_common_base_audit/checkpoint_inventory.json').write_text(json.dumps(report,indent=2)+'\n')
