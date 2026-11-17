"""Apply the official TIES-Merging implementation to URS checkpoints on CPU."""

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from third_party.ties_merging.minimal import (
    check_parameterNamesMatch,
    state_dict_to_vector,
    ties_merging,
    vector_to_state_dict,
)

UPSTREAM_REPOSITORY = "https://github.com/prateeky2806/ties-merging"
UPSTREAM_COMMIT = "44e7891fc84f3de7e4caa52664cd864ca3715e91"


def load_state(path):
    """Load an URS model state as CPU FP32 tensors."""
    with torch.serialization.safe_globals([set]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return {key: value.detach().float()
            for key, value in checkpoint["model_state_dict"].items()}


def prepare_vectors(base_path, checkpoint_paths):
    """Return the base state, base vector, and stacked expert-minus-base vectors."""
    base_state = load_state(base_path)
    states = [load_state(path) for path in checkpoint_paths]
    check_parameterNamesMatch([base_state] + states)
    for state in states:
        for key in base_state:
            if state[key].shape != base_state[key].shape:
                raise ValueError(f"Checkpoint parameter shape differs: {key}")
    base_vector = state_dict_to_vector(base_state)
    task_vectors = torch.stack([
        state_dict_to_vector(state) - base_vector for state in states
    ])
    return base_state, base_vector, task_vectors


def merge_from_vectors(base_state, base_vector, task_vectors,
                       keep_percent=20, scale=1.0):
    """Merge using official trimming, sign election, and disjoint mean."""
    if not 0 < keep_percent < 100:
        raise ValueError("keep_percent must be strictly between 0 and 100")
    # Convert explicit percentages to fractions, including 1% (official K=1
    # instead means 100%). The original threshold and tie handling stay intact.
    update = ties_merging(task_vectors, reset_thresh=keep_percent / 100,
                          merge_func="dis-mean")
    return vector_to_state_dict(base_vector + scale * update, base_state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--keep-percent", type=float, default=20)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    if not 0 < args.keep_percent < 100:
        parser.error("--keep-percent must be strictly between 0 and 100")
    base_state, base_vector, task_vectors = prepare_vectors(
        args.base, args.checkpoints)
    merged = merge_from_vectors(base_state, base_vector, task_vectors,
                                args.keep_percent, args.scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": merged,
        "merge_info": {
            "method": "ties-merging",
            "base": str(args.base.resolve()),
            "checkpoints": [str(path.resolve()) for path in args.checkpoints],
            "keep_percent": args.keep_percent,
            "scale": args.scale,
            "merge_func": "dis-mean",
            "dtype": "float32",
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
        },
    }, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
