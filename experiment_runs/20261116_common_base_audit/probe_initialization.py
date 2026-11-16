"""Reconstruct a seeded URS initialization and compare untouched parameter fingerprints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'URS'))
from model.Model import Model
from utils.utils import seed_everything

parser=argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu','cuda'], required=True)
parser.add_argument('--label', required=True)
parser.add_argument('--seed', type=int, default=3407)
args=parser.parse_args()
params=dict(embedding_dim=128,encoder_layer_num=12,ff_hidden_dim=512,
            logit_clipping=50,demand_max1=True,eval_type='greedy')
seed_everything(args.seed)
if args.device=='cuda':
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
model=Model(**params)
state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
fingerprints={k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k,v in state.items()}
inventory=json.loads((Path(__file__).resolve().parent/'checkpoint_inventory.json').read_text())
checks={}
for checkpoint,info in inventory.items():
    expected=info['unused_random_fingerprints']
    matched=[key for key,value in expected.items() if fingerprints[key]==value]
    checks[checkpoint]={'matched':len(matched),'total':len(expected),
                        'matched_numel':sum(state[key].numel() for key in matched),
                        'unmatched':[key for key in expected if key not in matched]}
    print(args.label,checkpoint,len(matched),'/',len(expected),flush=True)
metadata={
    'method':'reconstructed_random_initialization', 'seed':args.seed,
    'device':torch.cuda.get_device_name(0) if args.device=='cuda' else 'cpu',
    'torch':str(torch.__version__), 'cuda':torch.version.cuda,'model_params':params,
    'upstream_source_commit':'69c6766219e40d18fd8b371fedfc208241e5be9e',
    'historical_initial_checkpoint_obtained':False,'checks':checks,
    'parameter_hashes':fingerprints,
}
run=Path(__file__).resolve().parent
output=run/f'init_{args.label}_seed{args.seed}.pt'
torch.save({'model_state_dict':state,'reconstruction_info':metadata},output)
metadata['checkpoint_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Saved',output,flush=True)
