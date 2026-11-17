"""Official LOT closed-form task-vector fusion, adapted to URS parameter types.

Reference: third_party/lot_merging/methods/lot_merging_utils.py::lot_mergingv2.
Use complete expert features (feature_scale=1) and the upstream FP32 pinv.
"""

import torch

UPSTREAM_REPOSITORY = 'https://github.com/SunWenJu123/model-merging'
UPSTREAM_COMMIT = 'ab87731e405b9eeca1f822685ede0276d756907c'


def lot_vector(task_vectors, features, device='cuda'):
    """Return the optimal increment and diagnostics, before the final scale.

    Biases and unobserved layers retain the mean task increment. Linear layers
    solve pinv(sum X.T X) @ sum(X.T X @ delta.T), exactly as upstream. Norm
    scales use the elementwise squared-feature weighted average. Singular
    condition Grams are handled by the official Moore-Penrose pseudoinverse;
    no ridge, extra iterations, or null-space completion is added.
    """
    assert features['feature_scale'] == 1
    result = {name: (sum(tv[name] for tv in task_vectors) / len(task_vectors)).cpu()
              for name in task_vectors[0]}
    diagnostics = {'linear_layers': 0, 'norm_layers': 0, 'bias_tensors': 0,
                   'inactive_linear': [], 'hyper_gram_ranks': {},
                   'dtype': 'float32', 'solver': 'torch.linalg.pinv defaults'}
    for name, kind in features['parameter_kinds'].items():
        if kind == 'linear':
            gram_sum, weighted_sum = None, None
            for tv, feat in zip(task_vectors, features['grams']):
                if feat[name] is None:
                    continue
                gram = feat[name].to(device)
                weighted = gram @ tv[name].to(device).T
                gram_sum = gram if gram_sum is None else gram_sum + gram
                weighted_sum = weighted if weighted_sum is None else weighted_sum + weighted
            if gram_sum is None:
                diagnostics['inactive_linear'].append(name)
                continue
            result[name] = (torch.linalg.pinv(gram_sum) @ weighted_sum).T.cpu()
            diagnostics['linear_layers'] += 1
            if name.startswith('decoder.hyper_'):
                diagnostics['hyper_gram_ranks'][name] = {
                    'dimension': gram_sum.shape[0],
                    'rank': int(torch.linalg.matrix_rank(gram_sum).item()),
                }
        elif kind == 'norm':
            numerator, denominator = None, None
            for tv, feat, counts in zip(task_vectors, features['norm_sq'], features['counts']):
                if name not in feat:
                    continue
                squared_sum = feat[name] * counts[name]
                weighted = squared_sum * tv[name]
                numerator = weighted if numerator is None else numerator + weighted
                denominator = squared_sum if denominator is None else denominator + squared_sum
            if denominator is not None:
                denominator = torch.where(denominator == 0, torch.tensor(1e-10), denominator)
                result[name] = numerator / denominator
                diagnostics['norm_layers'] += 1
        else:
            assert kind == 'bias', (name, kind)
            diagnostics['bias_tensors'] += 1
    assert all(torch.isfinite(value).all() for value in result.values())
    diagnostics['merged_delta_norm'] = sum(
        value.double().square().sum().item() for value in result.values()) ** .5
    return result, diagnostics
