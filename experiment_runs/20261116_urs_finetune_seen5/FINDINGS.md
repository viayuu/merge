# URS 五任务微调结果

已完成五个独立专家的30个epoch微调，以及原URS和五个专家共6×5项评测。五项任务均为URS原已见训练任务：CVRP、CVRPB、CVRPTW、OCVRP、OCVRPTW。

所有专家从同一份多任务checkpoint出发；每epoch100000实例，AdamW，初始学习率3e-5，第26个epoch起为3e-6。每个专家共300万训练实例、23460次参数更新。CVRP与OCVRP在3090训练，其余在4090训练；所有最终评测统一使用4090、原URS数据及参考分数、N100、每任务1000例、greedy POMO、8倍增强。

**Gap单位为%，越低越好；Δ单位为百分点，新模型减原URS，负值表示改善。以下均为固定第30个epoch模型。**

| 模型 | CVRP | CVRPB | CVRPTW | OCVRP | OCVRPTW | 五任务平均 |
|---|---:|---:|---:|---:|---:|---:|
| 原URS多任务模型 | 1.812774 | 1.457159 | 6.129026 | 3.244275 | 5.072808 | 3.543209 |
| CVRP微调 | 1.680634 | 1.767453 | 6.337476 | 3.577126 | 5.409264 | 3.754391 |
| CVRPB微调 | 1.897565 | 1.322779 | 6.091157 | 3.270368 | 5.162637 | 3.548901 |
| CVRPTW微调 | 1.929292 | 1.641127 | 5.683701 | 3.439193 | 5.100948 | 3.558852 |
| OCVRP微调 | 2.334288 | 2.066791 | 6.271890 | 3.135779 | 5.319238 | 3.825597 |
| OCVRPTW微调 | 1.936522 | 1.819382 | 6.373931 | 3.572494 | 5.010008 | 3.742467 |

五个专家在各自微调任务上均优于原URS：

| 微调任务 | 原URS gap % | 新专家 gap % | 本任务Δ |
|---|---:|---:|---:|
| CVRP | 1.812774 | 1.680634 | -0.132140 |
| CVRPB | 1.457159 | 1.322779 | -0.134380 |
| CVRPTW | 6.129026 | 5.683701 | -0.445326 |
| OCVRP | 3.244275 | 3.135779 | -0.108496 |
| OCVRPTW | 5.072808 | 5.010008 | -0.062800 |

跨任务表现：20个非本任务组合中，19个gap上升，1个下降。唯一改善是CVRPB专家在CVRPTW上的gap下降0.037870个百分点；其余退化幅度为0.026094～0.609632个百分点，最大退化发生在OCVRP专家评测CVRPB时。

五个专家的五任务平均gap均高于原URS。CVRPB专家最接近原URS（平均Δ为+0.005693个百分点），其次是CVRPTW专家（+0.015644）；CVRP、OCVRP、OCVRPTW专家分别为+0.211182、+0.282389、+0.199259。本轮观察到专项提升与跨任务退化并存；尚未进行模型合并，不能据此声称合并有效。

产物核验：五条最终检查全部通过；均为epoch30、23460次优化器更新，权重全部有限，共同初始权重SHA一致。CSV的30条结果与六份原始评测JSON逐项相同，五个checkpoint文件SHA与最终核验记录相同。

数据与核验：[results.csv](/public/home/shiys/merge/experiment_runs/20261116_urs_finetune_seen5/results.csv)、[final_verification.json](/public/home/shiys/merge/experiment_runs/20261116_urs_finetune_seen5/final_verification.json)。

模型文件：

- [CVRP](/public/home/shiys/merge/experiment_runs/20261116_urs_finetune_seen5/models/cvrp/checkpoint_30.pt)
- [CVRPB](/public/home/shiys/merge/experiment_runs/20261116_urs_finetune_seen5/models/cvrpb/checkpoint_30.pt)
- [CVRPTW](/public/home/shiys/merge/experiment_runs/20261116_urs_finetune_seen5/models/cvrptw/checkpoint_30.pt)
- [OCVRP](/public/home/shiys/merge/experiment_runs/20261116_urs_finetune_seen5/models/ocvrp/checkpoint_30.pt)
- [OCVRPTW](/public/home/shiys/merge/experiment_runs/20261116_urs_finetune_seen5/models/ocvrptw/checkpoint_30.pt)
