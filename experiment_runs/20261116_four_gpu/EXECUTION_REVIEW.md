# Execution review

Overall judgment: **APPROVE — no blocking issue found in the revised per-node Slurm execution for this fixed manifest and the verified allocations.**

Scope: read-only review of `manifest.json`, `run_queue.py`, `run_task.sh`, and `gpu_test.py`; checked original argument defaults and JSON export contract. The project directory is not a Git repository, so review used the current file contents rather than a Git diff. No GPU work was launched by this review.

## Blocking issues

None found.

## Verified behavior

- The manifest contains 45 distinct checkpoint/task pairs, 45 distinct job IDs, output paths, and log paths. All checkpoint files exist. Phase counts are B1=12, B2=8, B3=25.
- The revised controller launches one Slurm step per node per phase, with `-c4`. Each step starts two local workers for that node's distinct GPU slots. This avoids requesting two simultaneous steps that each occupy the allocation's whole GPU set on this Slurm installation. Within one controller invocation, concurrency remains at most four and each assigned GPU runs at most one evaluation.
- Each node process enters only existing allocation 1475 or 1465 using `srun --jobid --overlap`. Each evaluation subprocess receives its individual `CUDA_VISIBLE_DEVICES` inside that step, then the original evaluator receives `--cuda 0`. The scripts contain no allocation cancellation, host-wide process termination, or host-direct GPU execution. The controller's Python environment is carried through its absolute `sys.executable`; launch it from `easynco` as planned.
- Evaluation uses `/public/home/shiys/merge/URS/dataset`, N=100, 1000 episodes, greedy decoding, seed 3407, augmentation 8/128, and original batch defaults `[1000, 125, 50, 8]`. The original `URS/test.py` is executed directly. The manifest explicitly records separate original subprocesses with the same seed per task.
- CUDA availability and exactly one visible device are required before entering the original evaluator; CPU fallback is rejected. Results must report a CUDA device.
- Nonzero evaluation subprocess exit becomes `failed`. Zero exit additionally requires readable JSON, no missing rows, exactly one row, matching task, 1000 evaluated episodes, a CUDA device, and finite objective/gap values.
- Each node writes a distinct state file; its two threads serialize updates with a lock and atomically replace the file. The controller alone writes `queue_state.json`, merging node records by job ID every five seconds. Exited node workers leave all unfinished assigned jobs failed. The controller waits for both steps and cannot advance unless all phase jobs are completed; analysis failure also stops progression.
- Base-dependent methods are recorded as `BASE_UNVERIFIED` and are absent from this batch.

## Non-blocking operational notes

- The running-state guard is intended for this single queue invocation. After an interrupted controller, inspect remaining Slurm steps before reconciling stale `running` state; do not start a competing queue. This is consistent with the requested small fixed-purpose implementation.
- The parent reports that sanity work has ended and no old steps remain. No additional GPU or test work was launched for this revision review.

No additional test suite or defensive framework is needed for the reviewed scope.
