"""Collect CAT/LOT layer statistics from original, unlabeled URS instances."""

import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'URS'))

from data.DataFinder import DataFinder
from data.DataReader import get_saved_data
from env.UniVRPEnv import UniVRPEnv
from model.Model import Model
from problem.ProblemRep import get_problem_representations


def collect_features(base_state, expert_states, tasks, sample_indices,
                     feature_scale, device='cuda', feature_batch_size=128):
    """Return CPU sufficient statistics, in the same order as ``tasks``.

    ``sample_indices`` maps each task to indices in its original 1000 instances.
    Linear weights use the unnormalized X.T @ X from the official CAT code.
    InstanceNorm affine scales use mean squared normalized channel features.
    The returned ``parameter_kinds`` distinguishes linear weights, norm scales,
    and biases by module type, not by substrings such as "bias" in URS names.
    """
    device = torch.device(device)
    metadata = ROOT / ('experiment_runs/20261116_urs_finetune_seen5/'
                       'models/cvrp/finetune_metadata.json')
    model_params = json.loads(metadata.read_text())['model_params']
    finder = DataFinder(data_dir=str(ROOT / 'URS/dataset'))
    representations = get_problem_representations()
    result = {
        'grams': [], 'norm_sq': [], 'counts': [], 'inactive': [],
        'parameter_kinds': {}, 'sample_indices': sample_indices,
        'feature_scale': feature_scale, 'tasks': list(tasks),
        'data_paths': {}, 'augmentation': 1, 'labels_used': False,
        'feature_forward': 'decoder.assign + model.pre_forward',
        'feature_batch_size': feature_batch_size,
    }

    for task, expert in zip(tasks, expert_states):
        model = Model(**model_params).to(device).eval()
        model.load_state_dict({
            name: base_state[name] + feature_scale * (expert[name] - base_state[name])
            for name in base_state
        }, strict=True)
        grams, norm_sums, counts = {}, {}, {}
        hooks = []

        def linear_hook(name):
            def collect(module, inputs, output):
                # Task conditions do not change across instance chunks. Count
                # their one forward input once, as in the unchunked path.
                if inputs[0].ndim == 1 and counts[name]:
                    return
                x = inputs[0].detach().reshape(-1, module.in_features)
                gram = (x.T @ x).cpu()
                grams[name] = gram if grams[name] is None else grams[name] + gram
                counts[name] += x.shape[0]
            return collect

        def norm_hook(name):
            def collect(module, inputs, output):
                # InstanceNorm stores channels on axis 1, unlike CLIP LayerNorm.
                normalized = ((output - module.bias[None, :, None]) /
                              module.weight[None, :, None])
                x = normalized.transpose(1, 2).reshape(-1, module.num_features)
                squared_sum = x.square().sum(0).cpu()
                norm_sums[name] = squared_sum if name not in norm_sums else (
                    norm_sums[name] + squared_sum)
                counts[name] += x.shape[0]
            return collect

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                key = name + '.weight'
                result['parameter_kinds'][key] = 'linear'
                grams[key], counts[key] = None, 0
                hooks.append(module.register_forward_hook(linear_hook(key)))
                if module.bias is not None:
                    result['parameter_kinds'][name + '.bias'] = 'bias'
            elif isinstance(module, nn.InstanceNorm1d):
                key = name + '.weight'
                result['parameter_kinds'][key] = 'norm'
                result['parameter_kinds'][name + '.bias'] = 'bias'
                counts[key] = 0
                hooks.append(module.register_forward_hook(norm_hook(key)))

        assert set(result['parameter_kinds']) == set(base_state)
        data_path = finder.get(task, 100)['data_path']
        result['data_paths'][task] = str(data_path)
        # Load only input data. No solution file or objective values enter CAT.
        with torch.device('cpu'):
            data, _ = get_saved_data(data_path, task, 1000, torch.device('cpu'),
                                     solution_name=None)
        indices = torch.tensor(sample_indices[task], dtype=torch.long, device='cpu')
        selected = {key: value[indices].to(device) if torch.is_tensor(value) else value
                    for key, value in data.items()}
        try:
            with torch.device(device), torch.inference_mode():
                env = UniVRPEnv()
                representation = torch.tensor(representations[task], dtype=torch.float32,
                                              device=device)
                # Every persistent decoder parameter is used in assign(); later
                # rollout F.linear calls use generated tensors, not parameters.
                model.decoder.assign(representation)
                for start in range(0, len(indices), feature_batch_size):
                    batch_size = min(feature_batch_size, len(indices) - start)
                    env.load_problems(batch_size=batch_size, problem_name=task,
                                      problem_size=100, pomo_size=100,
                                      validation_data=selected, aug_factor=1,
                                      start=start, device=device)
                    reset_state, _, _ = env.reset()
                    model.pre_forward(reset_state, task, representation)
        finally:
            for hook in hooks:
                hook.remove()

        norm_sq = {name: value / counts[name] for name, value in norm_sums.items()}
        assert all(torch.isfinite(value).all() for value in
                   [v for v in grams.values() if v is not None] + list(norm_sq.values()))
        result['grams'].append(grams)
        result['norm_sq'].append(norm_sq)
        result['counts'].append(counts)
        result['inactive'].append([name for name, count in counts.items() if count == 0])
        print(f'Merge features {task}: {len(indices)} instances, '
              f'{sum(v is not None for v in grams.values())} linear weights, '
              f'{len(norm_sq)} normalization scales', flush=True)
        del model, env
    return result
