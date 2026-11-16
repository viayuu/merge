# URS 单任务模型合并：研究简报

日期：2026-11-16（Asia/Shanghai）。以用户最新指令为准。

目标：复用URS已训练的11个单任务模型，经离线参数合并获得一个固定大小、能解多个任务的新模型，在原论文数据集与评测口径下对比通用URS及URS-STL，先判断motivation是否有信号。

数据约定：直接使用 `URS/dataset` 发布数据和原 `DataFinder`、`DataReader`、`Tester`；不另生成数据、不另划分calibration/test。保留原参考分数和gap公式，以论文Table 2/3的gap作为主要对照。正式N=100、每任务前1,000例、对称8倍/非对称128倍增强。小样本只用于代码自检。

本轮结果：代码、必要检查及45项正式评测均已完成（B1/B2/B3为12/8/25项，0失败），使用2×RTX4090和2×RTX3090。2/4/11专家等权合并的任务平均gap分别为26.085%/75.566%/21.026%，对应本地URS为3.971%/3.161%/2.734%；当前直接平均与所测插值未接近URS。详见[正式实验结果](refine-logs/EXPERIMENT_RESULTS.md)。共同base未核实，TA/TIES/DARE未运行，B4仍后置。

运行：默认 `conda activate easynco`；长任务放tmux；不设此前取消的pilot或GPU小时上限。当前shell位于admin01登录节点，实际通过已有Slurm allocation 1475（gpu05，2×RTX4090 24GB）和1465（gpu03，2×RTX3090 24GB）使用四张卡。每个allocation建立一个srun step，内部两个worker分别绑定两张卡；依据见 `refine-logs/GPU_ACCESS_AUDIT.md`。正式运行固定N=100、每任务1,000例、greedy、seed=3407、增强8/128，全程沿用原默认batch。

非目标：继续算子选择、重新找idea、设计新merge算法、用MoE/ensemble替代单一固定参数模型、为这轮代码建立新的数据或复杂评测基础设施。本次负结果限定于已测平均/插值配置。

Common Base最新状态：已完成[专项审计](experiment_runs/20261116_common_base_audit/RESULTS.md)，得到通过未更新随机参数指纹校验的两种CUDA初始化候选，专家统计上呈9+2组。允许同组、重建base的TA/TIES/DARE探索；没有获得作者完整起点谱系，也尚无新性能结果。
