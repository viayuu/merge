# URS 单任务模型合并：正式实验结果

2026-11-16。45/45项正式评测全部成功，使用2×RTX4090和2×RTX3090，正式任务执行时间为01:22:18—01:42:51（Asia/Shanghai）。B1/B2/B3分别完成12/8/25项，17个checkpoint中包含5个合并模型。

**结论：当前这批发布的URS单任务checkpoint，直接等权平均和已测试的三个线性插值配置均未接近通用URS。当前motivation尚未获得这组实验的正面支持。** 此结论限定于实际测试的配置，不代表所有模型合并方法。

**原论文结果能复现吗？**

基本吻合。本地11任务URS平均gap为2.734%，论文为2.735%；11个原专家在各自任务上的平均gap为1.738%，论文STL为1.740%。通用模型逐任务最大偏差为0.018163个百分点，专家最大偏差为0.010440个百分点，均出现在ACVRP。保留论文值与本地值，不把二者视为完全相同。

直接使用原URS/dataset、原DataReader/Tester/参考值/gap；N=100，每任务原顺序前1000例，greedy，seed3407，对称8倍/非对称128倍增强。全程沿用原batch [1000,125,50,8]，没有OOM降批、生成数据或重新拆分数据。原POMO规则得到CVRPB=80、PDTSP=50，其余=100。每个模型任务独立进程重置seed；非对称任务的随机node IDs仍由原代码生成。

**平均合并能接近URS吗？**

下表为相同任务集合内的等权平均gap（%，越低越好）；Δ为合并减本地URS，单位百分点。T2为CVRP/CVRPTW；T4额外包含CVRPB/OCVRP；T11为全部11项。

| 模型 | 任务集合 | 论文URS | 本地URS | 合并gap | Δ本地URS |
|---|---|---:|---:|---:|---:|
| 2专家等权 | T2 | 3.970 | 3.971 | 26.085 | +22.114 |
| 4专家等权 | T4 | 3.160 | 3.161 | 75.566 | +72.406 |
| 11专家等权 | T11 | 2.735 | 2.734 | 21.026 | +18.292 |

三种等权模型在各自评测集合内均明显退化。不能由这三行跨集合均值推导模型数量的单调效应：在共同T2上，2/4/11专家平均的gap依次为26.085%/70.896%/24.233%，并不随专家数单调增加。

**十一任务分别表现如何？**

| 任务 | 论文URS | 本地URS | 论文STL | 本地STL | 11专家合并 | Δ本地URS |
|---|---:|---:|---:|---:|---:|---:|
| TSP | 0.570 | 0.570 | 0.080 | 0.080 | 9.705 | +9.134 |
| ATSP | 2.260 | 2.258 | 0.620 | 0.620 | 21.348 | +19.089 |
| CVRP | 1.810 | 1.813 | 1.630 | 1.631 | 16.171 | +14.358 |
| ACVRP | 3.060 | 3.042 | 2.010 | 2.000 | 47.547 | +44.505 |
| CVRPB | 1.460 | 1.457 | 1.590 | 1.587 | 22.861 | +21.404 |
| CVRPTW | 6.130 | 6.129 | 4.500 | 4.498 | 32.296 | +26.167 |
| OCVRP | 3.240 | 3.244 | 2.250 | 2.247 | 18.087 | +14.843 |
| OCVRPTW | 5.070 | 5.073 | 3.220 | 3.222 | 22.780 | +17.708 |
| OP | 0.450 | 0.449 | 0.110 | 0.106 | 8.939 | +8.490 |
| PCTSP | 1.060 | 1.065 | -0.090 | -0.086 | 12.334 | +11.270 |
| PDTSP | 4.980 | 4.976 | 3.220 | 3.217 | 19.215 | +14.239 |

11专家合并在11/11任务均差于本地URS，也均差于对应原专家。原专家按任务分别使用，是11模型参考系统；不能将其平均成绩称为单个统一模型的成绩。PCTSP原参考口径产生的负gap按原值保留。

**调整两个专家的权重能解决吗？**

定义 θ(α)=(1−α)θCVRP+αθCVRPTW；每个α是一套固定参数，同时评测两个任务。0/1端点复用原专家结果。

| α | CVRP gap | CVRPTW gap | T2平均gap |
|---:|---:|---:|---:|
| 0 | 1.631 | 30.042 | 15.836 |
| 0.25 | 7.314 | 35.648 | 21.481 |
| 0.5 | 16.827 | 35.342 | 26.085 |
| 0.75 | 23.648 | 16.604 | 20.126 |
| 1 | 15.670 | 4.498 | 10.084 |

三个内部插值点均差于两个端点的任务平均成绩。内部点最优为α=.75，平均gap20.126%，仍远高于本地URS的3.971%。没有观察到平均或这组权重插值能保留两专家专长的证据；尚未确认的初始化关系、参数排列或模块兼容性只能作为后续待检验解释。

![原数据集上的插值与任务集合对照](/public/home/shiys/merge/experiment_runs/20261116_four_gpu/results_overview.png)

**哪些已完成，哪些仍有前提？**

本次计划内45项基础评测全部完成，无缺失、NaN或CPU回退。本次Slurm steps和tmux队列正常退出；保留用户原allocation1475/1465。两张4090分别完成12/11项，两张3090分别完成13/9项；利用率随导入、推理及结果写入变化，不代表持续满载。

TA/TIES/DARE的代码已实现，但真实共同base尚未核实，本次未运行这些方法，也未用通用URS充当base。99个未见任务按既有方案后置。继续当前merge方向的下一步应先确认专家训练起点及可用base，再评测这些有前提的方法；当前结果不支持直接扩大简单平均实验并宣称motivation成功。

完整逐项数值：[汇总](/public/home/shiys/merge/experiment_runs/20261116_four_gpu/summary.md)、[CSV](/public/home/shiys/merge/experiment_runs/20261116_four_gpu/results.csv)、[原始JSON目录](/public/home/shiys/merge/experiment_runs/20261116_four_gpu/results)、[PNG](/public/home/shiys/merge/experiment_runs/20261116_four_gpu/results_overview.png)、[PDF](/public/home/shiys/merge/experiment_runs/20261116_four_gpu/results_overview.pdf)。

运行与复核证据：[manifest](/public/home/shiys/merge/experiment_runs/20261116_four_gpu/manifest.json)、[队列状态](/public/home/shiys/merge/experiment_runs/20261116_four_gpu/queue_state.json)、[checkpoint来源及SHA256](/public/home/shiys/merge/experiment_runs/20261116_four_gpu/checkpoint_provenance.json)、[监控记录](/public/home/shiys/merge/experiment_runs/20261116_four_gpu/monitoring.md)、[结果复核](/public/home/shiys/merge/experiment_runs/20261116_four_gpu/RESULT_VERIFICATION.md)、[GPU实际采样](/public/home/shiys/merge/experiment_runs/20261116_four_gpu/gpu_utilization_snapshot.txt)。原始日志在同目录logs/，复现入口为launch.sh。

论文数字来源：arXiv:2509.23413v3，第6页Table 2/3；本地记录在merge_lab/paper_results.csv。
