# URS 单任务模型合并：研究简报

日期：2026-11-16（Asia/Shanghai）。以用户最新指令为准。

目标：复用URS已训练的11个单任务模型，经离线参数合并获得一个固定大小、能解多个任务的新模型，在原论文数据集与评测口径下对比通用URS及URS-STL，先判断motivation是否有信号。

数据约定：直接使用 `URS/dataset` 发布数据和原 `DataFinder`、`DataReader`、`Tester`；不另生成数据、不另划分calibration/test。保留原参考分数和gap公式，以论文Table 2/3的gap作为主要对照。正式N=100、每任务前1,000例、对称8倍/非对称128倍增强。小样本只用于代码自检。

本轮范围：按 `experiment-bridge` 完成合并、原评测调用和结果汇总代码，遵循KISS/YAGNI，做少量必要检查；不启动大规模实验、训练或新idea探索。共同base尚未核实，先支持平均/插值；TA/TIES/DARE在显式提供真实base时可用。

运行：默认 `conda activate easynco`；长任务放tmux；用户指定1×RTX4090 24GB；不设此前取消的pilot或GPU小时上限。当前shell尚未确认GPU驱动可见性，实施者不应把GPU环境问题混写成merge算法失败。

非目标：继续算子选择、重新找idea、设计新merge算法、用MoE/ensemble替代单一固定参数模型、为这轮代码建立新的数据或复杂评测基础设施。尚无merge性能结论。
