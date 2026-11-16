# Research Contract：URS单任务专家离线合并

日期：2026-11-16。用户固定idea及最新数据约定优先。

## 当前问题

只合并已发布URS单任务专家，得到一个固定参数量的多任务solver，在相同原论文数据和解码设置下接近或超过通用URS。先验证motivation，不重新寻找idea。

## 数据与证据

直接使用 `URS/dataset` 与原DataFinder/DataReader/Tester，不另生成、不另划分calibration/test，保留原参考分数和gap公式。正式N=100、前1,000实例/任务、对称8倍/非对称128倍增强，与arXiv v3 Table 2/3的URS及URS-STL对照。主输出为每任务原aug_gap及与论文的百分点差；少量实例只能作自检。

原计划的新held-out、paired bootstrap和相对退化阈值已由本次用户指令替换。若探索多个合并配置，保留完整结果并按同数据探索性比较呈现；不把论文结果、模型输出当作标签，不混写paper/local来源。

## 方法与实施边界

先均值/插值，真实base明确后再TA/TIES/DARE。通用URS只作对照；不同任务使用同一合并checkpoint与全局系数。保持原模型任务条件和mask。重新训练共享起点、动态专家、ensemble不是本轮工作。

当前按2→4→11任务的方案实现最少必要代码，遵循KISS/YAGNI，复用原评测，少量正确性检查即可；不新建数据或复杂测试框架，不自动启动大规模实验。用户指定1×RTX4090 24GB，无pilot/GPU小时上限，长任务放tmux。

本文件记录研究约定，不含实验成绩。下一步见 [EXPERIMENT_PLAN.md](../../refine-logs/EXPERIMENT_PLAN.md) 与 [EXPERIMENT_TRACKER.md](../../refine-logs/EXPERIMENT_TRACKER.md)。
