# URS Common Base 可行性审计

日期：2026-11-16（Asia/Shanghai）。本次任务是确认 Task Arithmetic、TIES、DARE 能否继续，不运行新的性能矩阵、不重训模型、不修改原数据集。结果见 [RESULTS.md](RESULTS.md)。

## 复现

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/merge
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python experiment_runs/20261116_common_base_audit/inspect_checkpoints.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python experiment_runs/20261116_common_base_audit/probe_initialization.py --device cpu --label cpu
srun --jobid=1475 --overlap -N1 -n1 -c2 --immediate=10 env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$CONDA_PREFIX/bin/python" experiment_runs/20261116_common_base_audit/probe_initialization.py --device cuda --label 4090
srun --jobid=1465 --overlap -N1 -n1 -c2 --immediate=10 env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$CONDA_PREFIX/bin/python" experiment_runs/20261116_common_base_audit/probe_initialization.py --device cuda --label 3090
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python experiment_runs/20261116_common_base_audit/compare_candidates.py
```

`1475`和`1465`是本次机器的已有allocation；在其他机器需替换入口。CPU与GPU probe均调用原Model与seed_everything；GPU初始化复用Trainer的默认CUDA张量设置和模型构造顺序。

## 产物

- `checkpoint_inventory.json`：12个原checkpoint的字段、optimizer state、未更新随机张量指纹。
- `init_{cpu,4090,3090}_seed3407.json`：候选初始化与各专家的精确指纹匹配、全部参数SHA256、环境和checkpoint哈希。
- `init_{cpu,4090,3090}_seed3407.pt`：重建候选，均由Git忽略；不冒充作者提供的原始epoch0文件。
- `gpu_initialization_difference.json`、`candidate_comparison.json`：两种CUDA初始化的区别、训练终点与候选的匹配及delta幅度。
- `implementation_check.json`：CVRP/OCVRP用候选base计算三种方法的有限性、strict-load检查；不是性能结果。

“匹配4090/3090候选”是本次重建CUDA初始化类型的简称，不能直接认定原训练物理GPU型号。分组依据包括未更新随机指纹和两候选差异坐标上的训练终点相关性；后者是强统计证据，尚无作者原始训练谱系记录。

## 独立复核

`verify_evidence.py`重新读取原checkpoint和候选，核对全部张量hash、未更新随机指纹、9＋2组，并额外检查4个诊断矩阵中是否还有精确未更新坐标；`verify_fresh_init.py --label 4090`或`--label 3090`应在对应GPU上执行，重新构造模型、校验382个张量；`plot_evidence.py`由独立复核JSON生成科学图。

本轮复核通过，但额外坐标检查没有找到能唯一证明4个历史初始矩阵的精确残留；因此没有把分组推断升级为完整历史base的确定性结论。
