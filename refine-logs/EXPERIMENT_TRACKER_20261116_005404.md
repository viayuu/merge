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
| 共同base | UNRESOLVED | 标准TA/TIES/DARE需显式提供真实base；未伪造 |
| GPU可见性 | UNAVAILABLE_IN_CURRENT_SHELL | torch CUDA=False、device_count=0；nvidia-smi无法连驱动 |
| B1两任务正式评测 | NOT_RUN | 已有average_2.pt及运行命令；需1000例、原增强 |
| B2四任务正式评测 | NOT_RUN | 同一合并/评测入口可直接运行 |
| B3十一任务正式评测 | NOT_RUN | 同一合并/评测入口可直接运行 |
| B4未见任务 | DEFERRED | 后续可用原发布数据及原评测；本轮不扩展 |

B0仅验证代码通路，没有正式motivation结论。日志为 `refine-logs/code_smoke.log`，退出码0；输出为 `results/code_smoke/`。首次源码备份错误和第二次缺openpyxl错误的日志保留为 `code_smoke_attempt1.log`、`code_smoke_attempt2.log`。
