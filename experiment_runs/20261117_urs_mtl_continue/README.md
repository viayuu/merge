# URS 多任务续训

两组模型均从 `unified_checkpoint_500.pt` 恢复模型与AdamW状态，追加30轮（501–530）。实际学习率1e-5、weight decay 1e-6、batch128；原451轮衰减已经执行，本次不另加衰减。任务为 CVRP、CVRPB、CVRPTW、OCVRP、OCVRPTW。

| 目录 | 4090槽位 | 每轮实例 | 30轮累计实例 |
| --- | --- | --- | --- |
| `models/total_100k/` | 0 | 五任务合计100,000 | 3,000,000 |
| `models/per_task_100k/` | 1 | 每任务100,000，共500,000 | 15,000,000 |

- [当前进度](PROGRESS.md)：队列每分钟自动更新，列出当前轮各任务实例数。
- [方案与原URS设置依据](PLAN.md) · [输入与代码记录](manifest.json) · [启动检查](smoke_verification.json)。
- [结果比较](RESULTS.md)：每组训练完成后自动运行原五任务完整评测并更新；未完成的组明确标为待完成。

正式运行位于 tmux 会话 `urs_mtl_continue`，使用 allocation1475/gpu05 的两张4090。`queue.log` 是调度日志，`logs/*_train.log` 是训练日志。每个模型目录保存 `metadata.json`、`progress.json`、逐轮 `training_counts.jsonl` 和checkpoint；最终使用 `checkpoint_530.pt`。

查看进度：

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/merge
python experiment_runs/20261117_urs_mtl_continue/progress.py
```

新实验在新目录运行，入口为 `launch.sh`，小规模检查加 `--smoke`。本轮正在运行时不要重复启动同一目录。异常会记录failed供检查，不会自动改配置重试。

此前五个专家使用重置优化器的3e-5→3e-6微调；本轮按用户要求采用原URS checkpoint续训语义。第二组匹配五专家累计训练实例与更新次数，优化器状态和学习率则不同。
