"""Recreate the candidate again in memory and verify every parameter tensor."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT / 'URS'))
from model.Model import Model
from utils.utils import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument('--label', choices=['cpu', '4090', '3090'], required=True)
args = parser.parse_args()
expected = json.loads((RUN / f'init_{args.label}_seed3407.json').read_text())
seed_everything(3407)
if args.label != 'cpu':
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.cuda.get_device_name(0)
    assert args.label in device
else:
    device = 'cpu'
model = Model(**expected['model_params'])
state = model.state_dict()
actual = {name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
          for name, value in state.items()}
assert actual == expected['parameter_hashes']
result = {'created': datetime.now().astimezone().isoformat(), 'device': device,
          'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'seed': 3407,
          'matched_tensors': len(actual), 'matched_coordinates': sum(v.numel() for v in state.values()),
          'all_tensor_hashes_exact': True, 'historical_epoch0_available': False}
(RUN / f'fresh_init_{args.label}_verification.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
