"""Merge URS checkpoints on CPU; the output loads directly in URS/test.py."""

import argparse
import math
from pathlib import Path

import torch


METHODS = ("average", "task-arithmetic", "ties", "dare-ta")


def load_state(path):
    # Published URS checkpoints include a set in result_log.
    with torch.serialization.safe_globals([set]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint["model_state_dict"]


def merge_states(states, method="average", *, base=None, weights=None,
                 scale=1.0, density=0.2, drop_rate=0.1, seed=0):
    """Merge matching state dicts in FP32 and return CPU tensors.

    `weights` are normalized averaging weights. Delta methods require the
    actual common base; this function cannot establish checkpoint lineage.
    TIES retains top-density coordinates across each whole task vector.
    """
    if not states or method not in METHODS:
        raise ValueError("Provide at least one state dict and a supported method")
    if method != "average" and base is None:
        raise ValueError(f"{method} requires an explicit common base checkpoint")
    if method != "average" and weights is not None:
        raise ValueError("weights are only used by average")
    if method == "average" and base is not None:
        raise ValueError("average does not use a base checkpoint")
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    if method == "ties" and not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    if method == "dare-ta" and not 0 <= drop_rate < 1:
        raise ValueError("drop-rate must be in [0, 1)")

    template = states[0]
    inputs = states + ([base] if base is not None else [])
    for state in inputs[1:]:
        if state.keys() != template.keys():
            raise ValueError("Checkpoint parameter keys differ")
        for key in template:
            if state[key].shape != template[key].shape:
                raise ValueError(f"Checkpoint parameter shape differs: {key}")

    float_keys = [key for key, value in template.items() if value.is_floating_point()]
    merged = {}
    for key in template.keys() - set(float_keys):
        if not all(torch.equal(template[key], state[key]) for state in inputs[1:]):
            raise ValueError(f"Non-floating values differ: {key}")
        merged[key] = template[key].detach().cpu().clone()

    def flatten(state):
        return torch.cat([state[key].detach().cpu().float().reshape(-1)
                          for key in float_keys])

    vectors = torch.stack([flatten(state) for state in states])
    if method == "average":
        coefficients = torch.tensor(weights if weights is not None else [1.] * len(states),
                                    dtype=torch.float32, device="cpu")
        if (coefficients.numel() != len(states) or not torch.isfinite(coefficients).all()
                or (coefficients < 0).any() or coefficients.sum() <= 0):
            raise ValueError("weights must match checkpoints, be nonnegative and have positive sum")
        coefficients = coefficients / coefficients.sum()
        result = (vectors * coefficients[:, None]).sum(dim=0)
    else:
        origin = flatten(base)
        deltas = vectors - origin
        if method == "ties":
            if density < 1:
                count = max(1, int(deltas.shape[1] * density))
                indices = deltas.abs().topk(count, dim=1, sorted=False).indices
                trimmed = torch.zeros_like(deltas)
                trimmed.scatter_(1, indices, deltas.gather(1, indices))
                deltas = trimmed
            elected = deltas.sum(dim=0).sign()
            selected = (deltas.sign() == elected) & (deltas != 0)
            # A zero sign (including an exact tie) contributes zero.
            update = (deltas * selected).sum(dim=0) / selected.sum(dim=0).clamp_min(1)
        else:
            if method == "dare-ta" and drop_rate > 0:
                generator = torch.Generator(device="cpu").manual_seed(seed)
                keep = torch.rand(deltas.shape, generator=generator, device="cpu") >= drop_rate
                deltas = deltas * keep / (1 - drop_rate)
            update = deltas.sum(dim=0)
        result = origin + scale * update

    offset = 0
    for key in float_keys:
        value = template[key]
        merged[key] = result[offset:offset + value.numel()].reshape(value.shape).to(value.dtype).clone()
        offset += value.numel()
    return {key: merged[key] for key in template}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", choices=METHODS, default="average")
    parser.add_argument("--base", type=Path, help="Actual common base for delta methods; no default")
    parser.add_argument("--weights", nargs="+", type=float, help="Averaging weights, normalized to sum to 1")
    parser.add_argument("--scale", type=float, default=1., help="Scale of the merged task delta")
    parser.add_argument("--density", type=float, default=.2, help="TIES retained fraction (default: 0.2)")
    parser.add_argument("--drop-rate", type=float, default=.1, help="DARE delta drop probability")
    parser.add_argument("--seed", type=int, default=0, help="DARE mask seed")
    args = parser.parse_args()

    if args.method != "average" and args.base is None:
        parser.error(f"--base is required for {args.method}; do not substitute URS generalist for an unknown base")
    states = [load_state(path) for path in args.checkpoints]
    base = load_state(args.base) if args.base is not None else None
    merged = merge_states(states, args.method, base=base, weights=args.weights,
                          scale=args.scale, density=args.density,
                          drop_rate=args.drop_rate, seed=args.seed)
    info = {
        "method": args.method,
        "checkpoints": [str(path.resolve()) for path in args.checkpoints],
        "base": str(args.base.resolve()) if args.base is not None else None,
        "weights": ([value / sum(args.weights) for value in args.weights]
                    if args.weights is not None else [1 / len(states)] * len(states))
                   if args.method == "average" else None,
        "scale": args.scale if args.method != "average" else None,
        "density": args.density if args.method == "ties" else None,
        "drop_rate": args.drop_rate if args.method == "dare-ta" else None,
        "seed": args.seed if args.method == "dare-ta" else None,
        "ties_zero_sign": "zero",
        "merge_dtype": "float32",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": merged, "merge_info": info}, args.output)
    print(f"Saved {args.method}: {len(states)} checkpoints -> {args.output}")


if __name__ == "__main__":
    main()
