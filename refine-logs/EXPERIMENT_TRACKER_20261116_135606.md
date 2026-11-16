# 实验追踪

日期：2026-11-16。当前约定：原URS数据、原评测、与原论文gap对照；不新建数据集。

| 阶段 | 状态 | 证据 / 下一动作 |
|---|---|---|
| 文献与原模型审计 | DONE | 12个模型同构，5篇核心PDF已下载 |
| 论文结果整理 | DONE | merge_lab/paper_results.csv：11任务，v3第6页Table 2/3 |
| 合并实现 | DONE | average/插值、TA、TIES、DARE+TA；真实两专家均值strict-load通过 |
| 原评测及汇总 | DONE | 原test.py可导出JSON；runner输出CSV/Markdown论文对照 |
| 代码审阅 | APPROVE | 两项汇总问题与相对路径问题已修复；源码备份修复也经复核 |
| B0 CPU通路检查 | PASS | 2模型×2任务，每组合2例；原数据、无增强；JSON/CSV完整 |
| 共同base | CANDIDATES_RECONSTRUCTED | 历史完整base未获证实；两种CUDA候选通过未更新随机指纹，统计上呈9+2组。可做同组重建base探索；不能混称11模型共享预训练base |
| GPU访问 | VERIFIED | admin01为登录节点；已有Slurm 1475/gpu05双4090、1465/gpu03双3090均可用；每allocation一个srun、内部双worker；见GPU_ACCESS_AUDIT.md |
| B1两任务正式评测 | COMPLETED | 12项：本地URS×2、两专家×2、三种插值×2；12/12成功 |
| B2四任务正式评测 | COMPLETED | 新增8项：四专家均值×4，新增两任务的URS与本任务专家；复用B1重叠结果 |
| B3十一任务正式评测 | COMPLETED | 新增25项：十一专家均值×11，余下7任务的URS与本任务专家；复用已有结果 |
| B4未见任务 | DEFERRED | 后续可用原发布数据及原评测；本轮不扩展 |

B0仅验证代码通路，没有正式motivation结论。日志为 `refine-logs/code_smoke.log`，退出码0；输出为 `results/code_smoke/`。首次源码备份错误和第二次缺openpyxl错误的日志保留为 `code_smoke_attempt1.log`、`code_smoke_attempt2.log`。

执行快照：2026-11-16 01:25:38 CST。`tmux urs_merge_20261116` 已启动45项唯一checkpoint×任务评测，启动时4项进入worker（导入阶段）。统一原数据N=100、每任务前1,000例、greedy、seed=3407、增强8/128，先沿用原默认batch。B1/B2/B3状态仅表示当前执行进度，尚无正式性能结论。

2026-11-16 01:33 CST：B1/B2共20项已完成且无失败；B3四卡运行中。T2等权模型gap26.085%、本地URS3.971%；T4等权模型75.566%、本地URS3.161%。当前直接均值结果为负，最终T11结论待全部结果到齐。

最终状态（2026-11-16 01:42:51）：45/45正式评测完成，0失败。T11等权合并gap21.026%，本地URS2.734%，差+18.292个百分点；11项均退化。完整分析见[EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)。队列正常退出，原allocation保留。

## ACVRP差异原因补充（2026-11-16）

已用原数据完成10次ACVRP对照复跑。原3090单任务进程结果可逐位重复；4090上同一进程ATSP→ACVRP得到URS 3.060408%、ACVRP专家2.006599%，保留两位正好是论文3.06%/2.01%。原程序全进程设seed一次，ATSP推进了后续ACVRP随机节点标识的序列；此前每任务独立进程重设seed改变了这一输入。仅推进等量随机数、不运行ATSP时，20批标识与所有分数逐位相同，确认随机序列机制。GPU型号另有较小影响。原数据、参考分数和此前45项结果均保留；当前负面合并结论不变。详情：[ACVRP专项排查](../experiment_runs/20261116_acvrp_audit/RESULTS.md)。

## Common Base可行性复核（2026-11-16）

本次已按seed3407和原模型构造顺序重建CPU、4090、3090初始化。两种GPU候选都精确匹配11专家所有未更新随机参数；但完整候选在4个decoder矩阵的348160个坐标上不同。训练终点在这些差异坐标的相关性强烈提示两组：A=TSP/ATSP/CVRP/ACVRP/CVRPB/OCVRP/OCVRPTW/OP/PDTSP（4090候选），B=CVRPTW/PCTSP（3090候选）。分组是强统计推断，不是作者完整谱系证明，也不等于确认历史物理训练GPU。

因此可继续“同组专家＋明确标注的重建随机初始化base”的TA/TIES/DARE探索；不能把全部11模型直接宣称为共享一个已验证预训练base的标准复现。建议先CVRP+OCVRP，再CVRP/CVRPB/OCVRP/OCVRPTW。TA在scale=1/K时等于已有平均，DARE先检查单专家低drop率。所有后续性能评测仍使用原数据、原gap；本次仅完成候选重建与加载/有限性检查，没有运行新性能矩阵。

完整证据与限制：[Common Base专项审计](../experiment_runs/20261116_common_base_audit/RESULTS.md)。历史45项结果与用户已有merge.py编辑均保留。
