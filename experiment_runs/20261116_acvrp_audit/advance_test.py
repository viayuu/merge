"""Run the original URS tester while recording its random node identifier stream."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import torch

root = Path(__file__).resolve().parents[2]
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
os.chdir(root / 'URS')
sys.path.insert(0, str(root / 'URS'))
import model.Model_LIB as library

original = library.unified_node_position_construction
trace = []
advanced = False

def fingerprint(tensor):
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()


def traced(problems, problem_name):
    global advanced
    if not advanced:
        assert problem_name == 'acvrp'
        for _ in range(20):
            torch.rand((50 * 128, 100, 1), device=problems.device)
        advanced = True
    rng_before = torch.cuda.get_rng_state()
    before = fingerprint(rng_before)
    features = original(problems, problem_name)
    trace.append({
        'task': problem_name, 'shape': list(features.shape),
        'rng_before': before, 'rng_after': fingerprint(torch.cuda.get_rng_state()),
        'rng_before_bytes': rng_before.tolist(), 'rng_after_bytes': torch.cuda.get_rng_state().tolist(),
        'features_sha256': fingerprint(features),
        'first_node_ids': features[0, :5, 0].cpu().tolist(),
    })
    return features

library.unified_node_position_construction = traced
print('Assigned CUDA device:', torch.cuda.get_device_name(0), flush=True)
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda,
      'TF32 matmul:', torch.backends.cuda.matmul.allow_tf32, flush=True)
output = Path(sys.argv[sys.argv.index('--output_json') + 1])
runpy.run_path(str(root / 'URS/test.py'), run_name='__main__')
output.with_suffix('.trace.json').write_text(json.dumps({
    'device_name': torch.cuda.get_device_name(0),
    'control': 'Advance CUDA RNG by the 20 ATSP node-ID calls only; no ATSP inference.',
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
    'calls': trace,
}, indent=2) + '\n')
