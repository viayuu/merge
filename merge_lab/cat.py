"""CAT task-vector trimming: published-code behavior and paper equations.

Reference: third_party/cat_merging/methods/cat_merging_utils.py::bal_mergingj.
URS uses [out,in] Linear weights and InstanceNorm rather than CLIP LayerNorm.
"""

import torch

UPSTREAM_REPOSITORY = 'https://github.com/SunWenJu123/model-merging'
UPSTREAM_COMMIT = 'ab87731e405b9eeca1f822685ede0276d756907c'


def conflict_basis(deltas, grams, protected, count, lambd, upstream=False):
    """Eigenvectors of sum_j D_j (C_i-lambda*C_j) D_j.T, j != i.

    Large hypernetwork outputs use an equivalent QR reduction when the chosen
    extreme eigenvalues are nonzero. This is checked for every actual call;
    selecting null-space directions would need an explicit full-space policy.
    FP64 eigensolves keep roundoff in rank-deficient matrices below the cutoff.
    """
    other = [j for j in range(len(deltas)) if j != protected]
    rows = deltas[0].shape[0]
    count = min(count, rows)
    matrices = [deltas[j].double() for j in other]
    q = None
    if rows > 1024:
        q, _ = torch.linalg.qr(torch.cat(matrices, dim=1), mode='reduced')
        matrices = [q.T @ value for value in matrices]
    size = matrices[0].shape[0]
    g = torch.zeros((size, size), dtype=torch.float64, device=deltas[0].device)
    for j, value in zip(other, matrices):
        g += value @ (grams[protected].double() - lambd * grams[j].double()) @ value.T
    eigenvalues, eigenvectors = torch.linalg.eigh((g + g.T) / 2)
    if upstream and not (eigenvalues > 0).any():
        return None
    if q is None:
        basis = eigenvectors[:, :count] if upstream else eigenvectors[:, -count:]
    else:
        # At least c signed eigenvalues ensure the chosen extreme subspace is
        # represented by Q, also after sequential projections from other tasks.
        tolerance = eigenvalues.abs().max() * 1e-10
        selected = torch.nonzero(eigenvalues < -tolerance if upstream else
                                 eigenvalues > tolerance).flatten()
        if selected.numel() < count:
            raise ValueError('QR CAT requires c nonzero extreme eigenvalues; '
                             f'found {selected.numel()}, requested {count}')
        selected = selected[:count] if upstream else selected[-count:]
        basis = q @ eigenvectors[:, selected]
    return basis.to(deltas[0].dtype)


def trim_vectors(task_vectors, features, *, linear_count=4, linear_lambda=5,
                 norm_count=4, norm_lambda=10, bias_count=4,
                 upstream=False, device='cuda'):
    """Return modified vectors and compact diagnostics; never update models.

    upstream=True preserves the release's ascending eigenvector selection,
    sequential norm restoration, name exclusions, and linear-only entry point.
    The paper path uses largest eigenvectors and Eqs. (14)/(17), without norm
    restoration. All bases/masks use the original, unmodified task vectors.
    """
    result = [{name: value.clone() for name, value in tv.items()}
              for tv in task_vectors]
    stats = {'linear_layers': 0, 'norm_layers': 0, 'bias_tensors': 0,
             'inactive_linear': [], 'upstream_name_exclusions': [],
             'zero_projection_count': 0, 'large_qr_layers': []}
    for name, kind in features['parameter_kinds'].items():
        original = [tv[name].to(device) for tv in task_vectors]
        if not any(value.count_nonzero() for value in original):
            continue
        if kind == 'linear' and linear_count:
            if upstream and 'bias' in name:
                stats['upstream_name_exclusions'].append(name)
                continue
            raw_grams = [feat[name] for feat in features['grams']]
            if any(value is None for value in raw_grams):
                # All five selected tasks leave the relation branch inactive.
                stats['inactive_linear'].append(name)
                continue
            grams = [value.to(device) for value in raw_grams]
            bases = [conflict_basis(original, grams, i, linear_count,
                                    linear_lambda, upstream)
                     for i in range(len(original))]
            for j, value in enumerate(original):
                if value.count_nonzero() == 0:
                    continue
                for i, basis in enumerate(bases):
                    if i == j or basis is None or basis.shape[1] == 0:
                        continue
                    projected = value - basis @ (basis.T @ value)
                    if upstream:
                        before, after = value.norm(), projected.norm()
                        if after > 0:
                            projected *= before / after
                        else:
                            stats['zero_projection_count'] += 1
                    value = projected
                result[j][name] = value.cpu()
            stats['linear_layers'] += 1
            if original[0].shape[0] > 1024:
                stats['large_qr_layers'].append(name)
        elif not upstream and ((kind == 'norm' and norm_count) or
                               (kind == 'bias' and bias_count)):
            count = min(norm_count if kind == 'norm' else bias_count,
                        original[0].numel())
            masks = []
            if kind == 'norm':
                sums = [feat[name].to(device) * counts[name]
                        for feat, counts in zip(features['norm_sq'], features['counts'])]
            for i in range(len(original)):
                score = torch.zeros_like(original[i])
                for j in range(len(original)):
                    if i != j:
                        score += original[j].square() * (
                            sums[i] - norm_lambda * sums[j] if kind == 'norm' else 1)
                mask = torch.ones_like(score)
                mask[torch.topk(score, count).indices] = 0
                masks.append(mask)
            for j, value in enumerate(original):
                for i, mask in enumerate(masks):
                    if i != j:
                        value = value * mask
                result[j][name] = value.cpu()
            stats['norm_layers' if kind == 'norm' else 'bias_tensors'] += 1
    for tv in result:
        assert all(torch.isfinite(value).all() for value in tv.values())
    stats['original_norms'] = [sum(v.double().square().sum().item()
                                  for v in tv.values()) ** .5 for tv in task_vectors]
    stats['trimmed_norms'] = [sum(v.double().square().sum().item()
                                 for v in tv.values()) ** .5 for tv in result]
    return result, stats


def merge_vectors(base, vectors, scale=0.4):
    """Official CAT final operation: base + alpha * sum(trimmed task vectors)."""
    return {name: value + scale * sum(tv[name] for tv in vectors)
            for name, value in base.items()}
