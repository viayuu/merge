# CSP + LOT source snapshot

Repository: https://github.com/SunWenJu123/model-merging

Fixed commit: `ab87731e405b9eeca1f822685ede0276d756907c`.

Reference files: `main_csp_lot.py`, `methods/rotate_ta.py`, `methods/lot_merging_utils.py`.

Paper: [ICML 2026, PMLR 306:116753–116775](https://proceedings.mlr.press/v306/sun26r.html).
Local PDF: `papers/csp_icml2026.pdf`, 691370 bytes, SHA256 `39961d327fe7c3f10796d46871fdb315fa1707c58a85e3b9d7b459c4058ab573`.

The entry computes LOT, applies `rotate_filter_pertv` with `parts=['outout']`, then scales the projected increment. ViT-B/32 settings are alpha=1.4, 64 exemplars, three repeats. The initial k=5 is overwritten by the actual sweep k=1,...,20. ViT-L/14 commented settings use alpha=1.6 and initial k=20.

The entry invokes `lot_merging`; the existing URS adapter uses the memory-efficient `lot_mergingv2`. Both implement the same closed-form linear/normalization fusion on this experiment's shared active layers. The v1 first-expert-missing-feature boundary does not occur for these five tasks. Both original functions were CPU-compared on the same synthetic Linear+LayerNorm inputs with exactly equal results. Existing URS module-type and full InstanceNorm-output adaptations are retained.

`rotate_filter_pertv` uses all 2D base tensors, FP32 economy SVD and the unrescaled out/out increment. Integer k is an absolute rank, clamped to each matrix's minimum dimension. Nonmatrix increments remain as LOT produces them. The 0<k<1 branch uses cumulative **squared** singular value energy; Appendix C.4 instead describes cumulative singular values in words. This run follows the published code and labels energy thresholds 0.1/0.5/0.9 as a URS extension, not published LOT defaults. The adapter's explicit k=0 identity bypass is a control: source k=0 would select at least one direction.

No repository license was declared upstream; no new license is assigned to these reference files.
