# 五个 URS 微调专家的 TIES 实验

输入为上一轮固定 epoch-30 的 CVRP、CVRPB、CVRPTW、OCVRP、OCVRPTW 专家；共同 base 为 `URS/pretrained/unified_checkpoint_500.pt`。所有输入、数据和输出模型的 SHA256 见 `manifest.json`。

- [方案](PLAN.md)：在首次结果产生前固定的66组TIES和6个对照。
- [结果](RESULTS.md)：完整比较、五专家交叉结果、最佳固定模型及原始结果链接。
- [完整扫描表](summary.csv) · [包含校准/复核的长表](all_results.csv) · [代表配置及模型路径](summary.json)。
- [实现验证](implementation_checks.json) · [跨卡校准](calibration.json) · [最终核验](final_verification.json)。

## 单个模型的复现

在项目根目录运行：

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
python merge_lab/ties.py \
  --base URS/pretrained/unified_checkpoint_500.pt \
  --checkpoints experiment_runs/20261116_urs_finetune_seen5/models/*/checkpoint_30.pt \
  --keep-percent 30 --scale 0.4 --output checkpoints/ties_k30_l0p4.pt
```

`keep-percent` 是名义保留百分数，官方阈值处的并列参数全部保留。核心函数来自固定版本的官方 notebook，见 [来源说明](../../third_party/ties_merging/README.md)。原 `merge.py --method ties` 的历史简化实现不用于本轮。

## 批量脚本

`build_models.py` 生成固定配置并写输入/模型 manifest 与评测计划；`check_calibration.py` 检查原 base 和3090/4090一致性；`run_evaluations.py` 读取 `evaluation_plan.json`，每节点两卡逐项调用原 URS 测试入口；`diagnose_vectors.py` 生成阈值与增量诊断；`analyze.py` 从已完成结果汇总。

```bash
python experiment_runs/20261117_ties_seen5/analyze.py --partial
# 全部完成后生成完整报告和图：
python experiment_runs/20261117_ties_seen5/diagnose_vectors.py
python experiment_runs/20261117_ties_seen5/analyze.py
```

全流程的 GPU 入口为 `bash launch_node.sh gpu05|gpu03 calibration|scan|confirm`。脚本中的 Slurm allocation 是本次运行时的1475/1465，应按实际环境调整。长任务置于 tmux。新一轮实验应使用新的实验目录，避免覆盖本轮模型、状态及结果。

运行中因gpu05进程启动与Excel导出较慢，已将16项未完成评测移至gpu03。`rebalance.json` 和 `rebalance_history/` 保留调度变更及中断尝试；模型、任务、数据、指标和参数网格均保持不变。只停止了本轮一个GPU评测step，原Slurm allocation保留。

模型与数据继续由 Git 忽略，报告、脚本和小型结果文件可以纳入版本控制。
