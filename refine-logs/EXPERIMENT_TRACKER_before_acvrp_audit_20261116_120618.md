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
| 共同base | UNRESOLVED | 未核实共同训练起点；本批不执行标准TA/TIES/DARE，也不以通用URS冒充base |
| GPU访问 | VERIFIED | admin01为登录节点；已有Slurm 1475/gpu05双4090、1465/gpu03双3090均可用；每allocation一个srun、内部双worker；见GPU_ACCESS_AUDIT.md |
| B1两任务正式评测 | COMPLETED | 12项：本地URS×2、两专家×2、三种插值×2；12/12成功 |
| B2四任务正式评测 | COMPLETED | 新增8项：四专家均值×4，新增两任务的URS与本任务专家；复用B1重叠结果 |
| B3十一任务正式评测 | COMPLETED | 新增25项：十一专家均值×11，余下7任务的URS与本任务专家；复用已有结果 |
| B4未见任务 | DEFERRED | 后续可用原发布数据及原评测；本轮不扩展 |

B0仅验证代码通路，没有正式motivation结论。日志为 `refine-logs/code_smoke.log`，退出码0；输出为 `results/code_smoke/`。首次源码备份错误和第二次缺openpyxl错误的日志保留为 `code_smoke_attempt1.log`、`code_smoke_attempt2.log`。

执行快照：2026-11-16 01:25:38 CST。`tmux urs_merge_20261116` 已启动45项唯一checkpoint×任务评测，启动时4项进入worker（导入阶段）。统一原数据N=100、每任务前1,000例、greedy、seed=3407、增强8/128，先沿用原默认batch。B1/B2/B3状态仅表示当前执行进度，尚无正式性能结论。

2026-11-16 01:33 CST：B1/B2共20项已完成且无失败；B3四卡运行中。T2等权模型gap26.085%、本地URS3.971%；T4等权模型75.566%、本地URS3.161%。当前直接均值结果为负，最终T11结论待全部结果到齐。

最终状态（2026-11-16 01:42:51）：45/45正式评测完成，0失败。T11等权合并gap21.026%，本地URS2.734%，差+18.292个百分点；11项均退化。完整分析见[EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)。队列正常退出，原allocation保留。
