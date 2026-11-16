# URS 模型合并正式评测

2026-11-16。用户授权执行既有方案并持续监控。使用 monitor-experiment / analyze-results 工作流，适配本机 Slurm 和 tmux。

- 输入与解码：原 `URS/dataset`，原 Tester/参考值/gap；N=100，每任务前1000例；greedy、原POMO多起点规则（CVRPB为80、PDTSP为50，其余为100），对称8倍/非对称128倍增强，seed3407，原batch [1000,125,50,8]。
- 每个模型任务由独立原test.py进程运行并重置同一seed；非对称任务随机node IDs受seed与batch影响。该方式不同于单进程顺序运行11任务时连续推进RNG；论文值与本地复跑分别报告。
- 45项：B1两任务12项，B2四任务8项增量，B3十一任务25项增量。重复基线复用相同结果；同一任务集合才计算宏均值。
- 5个新模型：两专家alpha=.25/.5/.75、四专家均值、十一专家均值。alpha定义为 (1-alpha)*CVRP + alpha*CVRPTW；端点为原专家。每个模型始终使用固定参数。
- 对照：通用URS全11任务、11个专家各自任务，以及CVRP/CVRPTW专家交叉评测。
- GPU：既有Slurm allocation1475/gpu05的2张RTX4090、1465/gpu03的2张RTX3090。每allocation单step，内部两个worker分别绑定分配范围内逻辑GPU0/1。admin01为登录节点，无CUDA不代表计算节点无卡。
- 原Slurm20.11的--overlap只共享CPU，未指定GRES的step占全部分配GPU，因此不使用每卡一个srun。开始两个sanity因step资源冲突失败，另外两个已成功；正式任务只使用修正后的调度结构。
- 共同base未知：TA/TIES/DARE尚不具备输入前提；未用通用URS冒充base。B4未见99任务按原方案后置。

主控tmux：`urs_merge_20261116`。`queue.log`为阶段日志；`logs/gpu*_B*_worker.log`为节点日志；`logs/<model>__<task>.log`为原评测日志。

实时状态：`queue_state.json`；原始结果：`results/*.json`；汇总：`results.csv`、`summary.md`。手动刷新：

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
python /public/home/shiys/merge/experiment_runs/20261116_four_gpu/analyze.py
```

队列遇到失败后不进入下一阶段。需先检查失败原因及Slurm steps；确认无旧worker运行后，才可 `bash launch.sh --retry-failed`。不能同时启动第二个队列。不要终止承载GPU的原allocation1465/1475。

小样本 `sanity_*.json` 仅用于运行检查，不纳入正式结果；正式结果与论文比较使用1000例。
