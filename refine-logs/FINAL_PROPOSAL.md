# 固定提案：复用URS单任务专家得到统一模型

日期：2026-11-16。以用户最新的数据与实施要求为准。

研究问题：将已发布11个URS单任务专家离线合成与URS同参数量的固定模型，在**URS原数据集与原评测口径**上接近或超过论文中的通用URS。

主轨先做参数平均与两模型插值，共同底座得到证实时再做Task Arithmetic、TIES、DARE。通用URS只作对照，保留原任务输入、条件模块、POMO和约束mask，不进行额外联合训练或按任务换模型。

评测直接复用 `URS/dataset`、原DataReader和Tester，不另生成或拆分数据。正式N=100、每任务前1,000例、对称8/非对称128增强；以原aug_gap与论文Table 2/3的URS及URS-STL gap比较，差值用百分点表示。本地通用模型复跑可作为补充，论文值与local值分开标注。

运行顺序仍为2→4→11源任务；少量原实例仅作代码自检。代码与必要检查已完成；现根据用户授权，使用2张RTX4090和2张RTX3090运行45项正式评测并监控分析。继续按KISS/YAGNI复用原代码与数据。结果与解释见[实验结果](EXPERIMENT_RESULTS.md)和[实时汇总](../experiment_runs/20261116_four_gpu/summary.md)。

最接近工作包括MPSF-FL和MoSES，不主张首次VRP merge；先取得现成专家一次离线合并的经验信号。共享base重训与99未见任务扩展后置。

执行依据：[实验方案](EXPERIMENT_PLAN.md)；状态：[实验追踪](EXPERIMENT_TRACKER.md)。
