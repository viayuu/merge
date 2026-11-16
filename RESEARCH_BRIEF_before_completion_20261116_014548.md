# URS 单任务模型合并：研究简报

日期：2026-11-16（Asia/Shanghai）。以用户最新指令为准。

目标：复用URS已训练的11个单任务模型，经离线参数合并获得一个固定大小、能解多个任务的新模型，在原论文数据集与评测口径下对比通用URS及URS-STL，先判断motivation是否有信号。

数据约定：直接使用 `URS/dataset` 发布数据和原 `DataFinder`、`DataReader`、`Tester`；不另生成数据、不另划分calibration/test。保留原参考分数和gap公式，以论文Table 2/3的gap作为主要对照。正式N=100、每任务前1,000例、对称8倍/非对称128倍增强。小样本只用于代码自检。

本轮范围：代码和必要检查已完成，现按用户授权执行2→4→11任务正式实验并持续监控、分析结果，遵循KISS/YAGNI。45项互不重复的checkpoint×任务评测已在 `tmux urs_merge_20261116` 启动：B1共12项为RUNNING，B2共8项与B3共25项为QUEUED；B4后置。共同base尚未核实，本批只执行平均/插值，不执行TA/TIES/DARE。

运行：默认 `conda activate easynco`；长任务放tmux；不设此前取消的pilot或GPU小时上限。当前shell位于admin01登录节点，实际通过已有Slurm allocation 1475（gpu05，2×RTX4090 24GB）和1465（gpu03，2×RTX3090 24GB）使用四张卡。每个allocation建立一个srun step，内部两个worker分别绑定两张卡；依据见 `refine-logs/GPU_ACCESS_AUDIT.md`。正式运行固定N=100、每任务1,000例、greedy、seed=3407、增强8/128，先沿用原默认batch。

非目标：继续算子选择、重新找idea、设计新merge算法、用MoE/ensemble替代单一固定参数模型、为这轮代码建立新的数据或复杂评测基础设施。尚无merge性能结论。
