# URS 原五任务单任务微调

用户已取消前一轮 CEMM 五任务组合。本轮仅训练并评测 CVRP、CVRPB、CVRPTW、OCVRP、OCVRPTW，均为 URS 原11训练任务中的任务，且原仓库均提供单任务专家。旧run `20261116_urs_finetune_cemm5` 已停止；其结果和中间权重不作为本轮输入。

- 五个专家均从 `URS/pretrained/unified_checkpoint_500.pt` 独立重新开始；只加载模型权重，优化器与epoch重置。
- 每任务30个epoch，每epoch精确100000例；batch128，781个完整batch及32例尾batch，共782次更新。每任务累计3000000例、23460次更新。
- AdamW，学习率3e-5，weight_decay=1e-6；epoch26开始学习率降至3e-6。seed3407。
- 复用URS原在线数据生成器、N100和POMO REINFORCE训练逻辑，不新增或划分评测数据。
- 每epoch仅验证本任务原1000例，采用原Trainer无增强验证；最终固定取epoch30，另用原Tester对五项任务交叉评测。
- 最终6模型×5任务：原数据各1000例、greedy、原POMO规则、8倍增强、batch125、固定任务顺序和seed。只比较新专家与原多任务模型；本轮不做merge或额外多任务训练对照。
- GPU：allocation1475/gpu05的2张RTX4090，加allocation1465/gpu03的2张RTX3090。每节点一个Slurm step、内部两worker。环境沿用easynco。

入口：

```bash
bash /public/home/shiys/merge/experiment_runs/20261116_urs_finetune_seen5/launch.sh
```

`--smoke`使用每任务1epoch、160训练例、16验证例，只检查执行通路。正式模型输出为 `models/<task>/checkpoint_30.pt`；元数据记录共同起点SHA。

`queue_state.json`和`logs/`记录执行状态；`PROGRESS.md`每分钟刷新；`RESULTS.md`和`results.csv`在每个模型完成五任务评测后自动更新。全部完成后`final_verification.json`核验5模型的epoch、更新次数、学习率、共同起点及有限权重。

正式启动：2026-11-16T22:47:19+08:00，tmux `urs_ft_seen5`，监控窗口每分钟更新进度。五任务smoke全部通过，记录在`smoke_verification.json`。

## 四卡调度更新

用户于2026-11-16追加授权2张3090。保留已运行的CVRPTW、OCVRPTW及其optimizer/RNG进度；2026-11-16 23:25开始在两张3090提前训练CVRP、OCVRP。CVRPB交给最先空闲的4090。

- 4090主控仍为tmux `urs_ft_seen5` / `queue_state.json`，负责所有最终五任务评测。主队列内存中的pending列表不被外部改写。
- 3090训练为tmux `urs_ft_seen5_3090` / `3090_state.json`，入口`launch_3090.sh`。它只训练，不写原主队列状态。
- 原4090队列轮到已提前训练的专家时，等待其训练完成并直接复用，随后在4090做评测，不再训练一遍。
- 同一输出目录通过原子mkdir `.training_claim`互斥；共享盘flock经跨节点实测不互斥，已弃用。原子mkdir已在gpu03/gpu05验证。正常退出释放claim；异常退出留下claim时需先确认无训练进程，再人工处理，禁止盲重启。
- completed复用检查task、base文件SHA、epoch、实例数、batch、学习率、衰减节点、seed及最终checkpoint。存在未完成metadata时禁止直接覆盖。
- `PROGRESS.md`叠加两节点状态，读取模型目录的`run.log`；3090 stdout为`logs/<task>_train_3090.log`，不会被主队列稍后的复用调用截断。
- 3090两任务通路检查已验证反向传播、保存、两次更新及有限权重；初始全部参数SHA与运行中的4090专家完全一致。详见`smoke_3090_verification.json`。
