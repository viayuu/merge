"""One numerical check against unmodified upstream algebra and dense CAT."""

import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT))
from merge_lab.cat import conflict_basis, trim_vectors


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    name = 'layer.weight'
    xs = [torch.randn(11, 8) for _ in range(5)]
    vectors = [{name: torch.randn(16, 8)} for _ in range(5)]
    features = {'parameter_kinds': {name: 'linear'},
                'grams': [{name: x.T @ x} for x in xs]}
    ours, _ = trim_vectors(vectors, features, linear_count=4, linear_lambda=.2,
                          upstream=True, device='cpu')
    path = ROOT / 'third_party/cat_merging/methods/cat_merging_utils.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'bal_mergingj')
    block = copy.deepcopy(next(node for node in function.body if isinstance(node, ast.If)
                               and ast.unparse(node.test) == 'args.ratio != 0'))
    # The published mathematical block is unchanged; only its device is CPU.
    for node in ast.walk(block):
        if isinstance(node, ast.Constant) and node.value == 'cuda':
            node.value = 'cpu'
    reference = [SimpleNamespace(vector=copy.deepcopy(value)) for value in vectors]
    namespace = {'torch': torch, 'np': np,
                 'args': SimpleNamespace(ratio=4, lambd=.2),
                 'feat_arr': [{name: x} for x in xs], 'result_task_vectors': reference,
                 'model': SimpleNamespace(named_parameters=lambda: [(name, None)])}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])),
                 str(path), 'exec'), namespace)
    source_error = max((a[name] - b.vector[name]).abs().max().item()
                       for a, b in zip(ours, reference))
    assert source_error < 2e-4, source_error

    # Force the large-output QR path, compare its action to a dense eigensolve.
    ds = [torch.randn(1040, 4, dtype=torch.float64) for _ in range(5)]
    xs = [torch.randn(6, 4, dtype=torch.float64) for _ in range(5)]
    grams = [x.T @ x for x in xs]
    g = sum(ds[j] @ (grams[0] - .7 * grams[j]) @ ds[j].T for j in range(1, 5))
    values, eigenvectors = torch.linalg.eigh((g + g.T) / 2)
    qr_errors = {}
    for upstream in [False, True]:
        basis = conflict_basis(ds, grams, 0, 4, .7, upstream)
        dense = eigenvectors[:, :4] if upstream else eigenvectors[:, -4:]
        error = max((basis @ (basis.T @ d) - dense @ (dense.T @ d)).abs().max().item()
                    for d in ds[1:])
        assert error < 1e-8, error
        qr_errors['upstream' if upstream else 'paper'] = error

    # Mask application must use OTHER tasks' scores and compound across i.
    vs = [{'shift.bias': torch.randn(8)} for _ in range(5)]
    expected = copy.deepcopy(vs)
    for i in range(5):
        score = sum(vs[j]['shift.bias'].square() for j in range(5) if j != i)
        removed = torch.topk(score, 2).indices
        for j in range(5):
            if i != j:
                expected[j]['shift.bias'][removed] = 0
    got, _ = trim_vectors(vs, {'parameter_kinds': {'shift.bias': 'bias'}},
                          linear_count=0, norm_count=0, bias_count=2, device='cpu')
    assert all(torch.equal(a['shift.bias'], b['shift.bias']) for a, b in zip(got, expected))
    report = {'upstream_linear_max_abs_error': source_error,
              'qr_dense_projected_delta_max_abs_errors': qr_errors,
              'paper_bias_other_task_masks': True, 'passed': True}
    (RUN / 'implementation_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
