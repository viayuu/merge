# TIES-Merging official minimal implementation

Source: [prateeky2806/ties-merging](https://github.com/prateeky2806/ties-merging),
commit [`44e7891fc84f3de7e4caa52664cd864ca3715e91`](https://github.com/prateeky2806/ties-merging/tree/44e7891fc84f3de7e4caa52664cd864ca3715e91).
The original notebook and supporting source files are archived under `src/`.
The upstream BSD 3-Clause license is in [LICENSE.txt](LICENSE.txt).

`minimal.py` extracts these nine functions from `src/ties_minimal.ipynb` without
changing their bodies: `state_dict_to_vector`, `vector_to_state_dict`,
`check_parameterNamesMatch`, `check_state_dicts_equal`, `topk_values_mask`,
`resolve_zero_signs`, `resolve_sign`, `disjoint_merge`, and `ties_merging`.
Only the required imports (`copy`, `OrderedDict`, `torch`) are included;
the notebook's model downloads and example execution are omitted.

The URS adapter is `merge_lab/ties.py`. It forms CPU FP32 task vectors relative
to the shared base, uses the official `dis-mean` implementation, and reconstructs
`base + scale * merged_task_vector`. All URS state keys are retained; ordering
and reconstruction use the original helpers.

```bash
python -m merge_lab.ties \
  --base URS/pretrained/unified_checkpoint_500.pt \
  --checkpoints expert1.pt expert2.pt \
  --keep-percent 20 --scale 1 --output merged.pt
```

`--keep-percent` is strictly between 0 and 100 and is converted to a fraction
before calling the official code. Its `kthvalue` threshold and `>=` comparison
are preserved, so ties can retain more than the nominal percentage. The
official majority-sign rule for zero coordinates is also unchanged. A scale
of zero recovers the base. The output contains `model_state_dict` and source
paths, hyperparameters, and the pinned upstream commit in `merge_info`.
