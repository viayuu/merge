> 已按用户要求于 2026-11-16T22:45:06+08:00 停止此旧任务组。后续实验为 `../20261116_urs_finetune_seen5/`，旧结果不纳入新组。

# URS 多任务共同起点 → CEMM 五任务专家

用户授权：五个任务各微调30个epoch，使用两张4090，评测五个新模型及原URS在这五个任务上的表现。本轮不进行merge、不评测11任务、不增加继续多任务训练对照。

| CEMM任务 | URS任务 | 原URS是否训练过 |
|---|---|---|
| CVRP | cvrp | 是 |
| OVRP | ocvrp | 是 |
| VRPB | cvrpbp | 否 |
| VRPL | cvrpl | 否 |
| VRPTW | cvrptw | 是 |

VRPB要求linehaul先于backhaul；URS的cvrpb允许交错，因此使用cvrpbp。五任务约束对应CEMM，数据与评价采用URS原协议。

- Base：`URS/pretrained/unified_checkpoint_500.pt`，五个专家各自从这份完整权重启动；只恢复模型，优化器重新创建。
- 全参数微调：AdamW，学习率3e-5，weight_decay=1e-6，epoch26开始降为3e-6，seed3407。
- 每任务30个epoch，每epoch精确100000例（对齐CEMM的epoch数据量）；batch128，781个完整batch及最后32例，共782步。每任务累计3000000例、23460步。
- N=100，原URS在线训练生成器及POMO REINFORCE目标。不新增评测集或划分。
- 每epoch在本任务原1000例上记录无增强验证值；最终结果固定使用epoch30，不按验证成绩挑选epoch。
- 最终6个模型×5任务：原URS Tester，N100、原前1000例、greedy、POMO多起点、8倍增强，batch125。五任务使用相同顺序及seed。直接与同次评测的base比较。
- 2张RTX4090：现有Slurm allocation1475/gpu05，单step内部2个worker分别绑定逻辑GPU0/1。不要终止原allocation。
- 环境沿用easynco，不安装或升级依赖。新入口先对五个任务各做1epoch160训练例、16验证例的通路检查；smoke结果不纳入正式比较。

正式入口：

```bash
bash /public/home/shiys/merge/experiment_runs/20261116_urs_finetune_cemm5/launch.sh
```

通路检查追加`--smoke`。长任务从tmux运行；日志在`queue.log`及`logs/`，状态在`queue_state.json`，专家在`models/<task>/checkpoint_30.pt`。

各专家的`finetune_metadata.json`记录共同base及初始权重校验、超参数。`evaluation/<model>/*.json`保留原评测明细；`RESULTS.md`和`results.csv`由`analyze.py`生成五任务矩阵及相对base差值。

实际启动：2026-11-16 22:28（Asia/Shanghai），tmux `urs_ft_cemm5`。同session的`monitor`窗口每分钟运行`progress.py`，更新`PROGRESS.md`；训练窗口退出后也会结束监控。

五任务smoke已全部通过：每模型epoch1、两次优化器更新（128+32例）、权重有限，五个起点的全部参数SHA一致。记录为`smoke_verification.json`；代码也经独立只读审查。完整实验结束后自动生成`final_verification.json`，核验epoch30、23460次更新、末期学习率、权重有限和共同起点。
