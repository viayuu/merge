"""Use the original URS entrypoint, requiring the assigned CUDA device."""
import os
from pathlib import Path
import runpy
import sys

import torch

assert torch.cuda.is_available(), 'This experiment requires CUDA; CPU fallback is disabled.'
assert torch.cuda.device_count() == 1, 'Bind exactly one assigned GPU before running.'
print('Assigned CUDA device:', torch.cuda.get_device_name(0), flush=True)
urs = Path(__file__).resolve().parents[2] / 'URS'
os.chdir(urs)
sys.path.insert(0, str(urs))
sys.argv[0] = str(urs / 'test.py')
runpy.run_path(sys.argv[0], run_name='__main__')
