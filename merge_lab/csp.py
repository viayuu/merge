"""CSP's official out/out projection, applied after LOT fusion.

Reference: third_party/csp_merging/methods/rotate_ta.py::rotate_filter_pertv.
Only the base SVD is cached; the rank rule and FP32 projection are unchanged.
"""

import torch

UPSTREAM_REPOSITORY = 'https://github.com/SunWenJu123/model-merging'
UPSTREAM_COMMIT = 'ab87731e405b9eeca1f822685ede0276d756907c'


@torch.inference_mode()
def base_svd(base, device='cuda'):
    """Return CPU (U, S, Vh) tuples for every two-dimensional base tensor."""
    result = {}
    for name, weight in base.items():
        if weight.ndim == 2:
            factors = torch.linalg.svd(weight.to(device=device, dtype=torch.float32),
                                       full_matrices=False)
            result[name] = tuple(value.cpu() for value in factors)
    return result


@torch.inference_mode()
def project_vector(delta, svd, k, device='cuda'):
    """Keep (I-U_r U_r.T) delta (I-V_r V_r.T); leave nonmatrices unchanged.

    As upstream, integer k >= 1 protects min(k, min(shape)) directions, -1
    protects all directions, and 0 < k < 1 selects cumulative squared singular
    value energy separately for each layer. There is no rescaling after CSP.
    k=0 is our explicit identity control, not upstream's energy branch (which
    would select one direction even at zero). Full-rank cancellation retains
    the upstream FP32 roundoff rather than forcibly replacing values with zero.
    """
    assert k in (-1, 0) or 0 < k < 1 or (k >= 1 and int(k) == k)
    result = {name: value.detach().cpu() for name, value in delta.items()}
    diagnostics = {'k': k, 'parts': ['outout'], 'dtype': 'float32',
                   'identity_control': k == 0, 'layers': {},
                   'unchanged_nonmatrix_tensors': sum(value.ndim != 2 for value in delta.values())}
    for name, (u, singular_values, vh) in svd.items():
        tv = delta[name].to(device=device, dtype=torch.float32)
        energy = singular_values.to(device).square()
        cumulative = energy.cumsum(0)
        total = cumulative[-1].clamp_min(1e-12)
        m = len(singular_values)
        if k == 0:
            rank = 0
        elif k == -1:
            rank = m
        elif k < 1:
            rank = min(int(torch.searchsorted(cumulative, k * total).item()) + 1, m)
        else:
            rank = min(int(k), m)

        if rank:
            ur = u[:, :rank].contiguous().to(device)
            vr = vh[:rank, :].T.contiguous().to(device)
            puw = ur @ (ur.T @ tv)
            wpv = (tv @ vr) @ vr.T
            in_in = ur @ (ur.T @ tv @ vr) @ vr.T
            projected = tv - puw - wpv + in_in
            result[name] = projected.cpu()
        else:
            projected = tv

        before = tv.double().norm().item()
        after = projected.double().norm().item()
        diagnostics['layers'][name] = {
            'shape': list(tv.shape), 'rank': rank, 'max_rank': m,
            'protected_energy_fraction': (energy[:rank].sum() / total).item(),
            'retained_delta_norm_ratio': after / before if before else 0.,
        }
    diagnostics['projected_matrices'] = len(svd) if k != 0 else 0
    before_sq = sum(value.double().square().sum().item() for value in delta.values())
    after_sq = sum(value.double().square().sum().item() for value in result.values())
    diagnostics['delta_norm_before'] = before_sq ** .5
    diagnostics['delta_norm_after'] = after_sq ** .5
    diagnostics['retained_delta_norm_ratio'] = (after_sq / before_sq) ** .5 if before_sq else 0.
    return result, diagnostics
