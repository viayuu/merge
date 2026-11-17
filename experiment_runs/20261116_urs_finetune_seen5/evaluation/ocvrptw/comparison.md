# URS 原数据集结果对比

Gap 沿用原 DataReader/Tester 的参考值及公式；差值单位为百分点，负值更好。
仅原 dataset、N=100、1000例、greedy 与论文增强倍数匹配的行计算论文差值。
小样本/无增强仅用于检查通路，不视为论文成绩复现。论文数值保留原表精度。

| 运行/模型 | 任务 | N | 实例数 | 增强 | 本次 gap % | 论文 URS % | 论文 STL % | ΔURS pp | ΔSTL pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ocvrptw/checkpoint_30 | cvrp | 100 | 1000 | 8 | 1.937 | 1.810 | 1.630 | 0.127 | 0.307 |
| ocvrptw/checkpoint_30 | cvrpb | 100 | 1000 | 8 | 1.819 | 1.460 | 1.590 | 0.359 | 0.229 |
| ocvrptw/checkpoint_30 | cvrptw | 100 | 1000 | 8 | 6.374 | 6.130 | 4.500 | 0.244 | 1.874 |
| ocvrptw/checkpoint_30 | ocvrp | 100 | 1000 | 8 | 3.572 | 3.240 | 2.250 | 0.332 | 1.322 |
| ocvrptw/checkpoint_30 | ocvrptw | 100 | 1000 | 8 | 5.010 | 5.070 | 3.220 | -0.060 | 1.790 |

## 同一次运行、同一规模内的任务等权均值

不同运行分别显示；比较模型时应使用相同任务集合。

| 运行/模型 | N | 任务集合 | 本次平均 gap % | 可与论文比较数 | 平均 ΔURS pp |
|---|---:|---|---:|---:|---:|
| ocvrptw/checkpoint_30 | 100 | cvrp, cvrpb, cvrptw, ocvrp, ocvrptw | 3.742 | 5 | 0.200 |

论文来源：arXiv:2509.23413v3，Table 2/3；详见 merge_lab/paper_results.md。
