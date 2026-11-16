# 实验追踪

日期：2026-11-16。所有性能结果为空，未启动GPU任务。

| 阶段 | 状态 | 已有证据 / 下一动作 |
|---|---|---|
| 文献定向调研 | DONE | 经典5篇PDF、routing近邻与源链接已保存 |
| 本地结构审计 | DONE | 12个checkpoint strict load，结构完全兼容 |
| 共同base确认 | UNRESOLVED | checkpoint无谱系；相同闲置参数不足以确认 |
| 评测/数据wrapper | PLANNED | 独立生成split，逐例目标/路线/合法性，固定随机输入 |
| GPU可见性 | UNVERIFIED | 当前shell nvidia-smi驱动通信失败；未修环境 |
| B0运行自检 | NOT_RUN | 核验数据、目标方向、解码、merge恒等性质 |
| B1两专家 | NOT_RUN | calibration上的U/专家/均值/插值 |
| B2四专家 | NOT_RUN | 固定任务集；候选与最佳单专家对照 |
| B3十一专家 | NOT_RUN | 冻结候选后统一揭盲独立test |
| B4九十九未见任务 | DEFERRED | B3有信号后；多仓数据单独处理 |

结果目录、运行日志、指标表在实施后填写；没有数字代表结果尚不存在，不能用空白解释为0或失败。
