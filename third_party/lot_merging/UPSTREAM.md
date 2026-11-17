# LOT Merging source snapshot

Repository: https://github.com/SunWenJu123/model-merging

Commit: `ab87731e405b9eeca1f822685ede0276d756907c` (verified current main).

Reference files: `main_lot.py`, `methods/lot_merging_utils.py`. These upstream reference files are preserved; the URS adapter is `merge_lab/lot.py` and shared feature collection is `merge_lab/cat_features.py`.

Paper: https://arxiv.org/html/2505.23859v2

NeurIPS 2025: https://papers.nips.cc/paper_files/paper/2025/hash/fbb4f2269267de23e0240e1c01242981-Abstract-Conference.html

Local proceedings PDF: `papers/lot_merging_neurips2025.pdf` (404114 bytes; SHA256 `459aa3f6436a5971fc3d4e5b182442b9f0e918edcff9e4928f35f09af90ed9d4`). Figure 3(b) scans scales 1.0 through 2.0 in steps of 0.1. This URS run keeps the released 1.2 default and uses a smaller exploratory sensitivity grid, including 0.6/0.8 outside that paper range.

The entry script sets `repeat=1`; the paper reports averages over three random exemplar selections. This URS study uses three selections for subsets and one fixed ordering for the full existing 1000-instance set. The source extracts exemplars from training data; this project uses the existing original evaluation inputs, as requested by the user, and reports that protocol difference.

Upstream declares no repository license; no new license is assigned to the reference snapshot.
