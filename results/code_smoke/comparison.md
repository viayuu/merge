# URS 原数据集结果对比

Gap 沿用原 DataReader/Tester 的参考值及公式；差值单位为百分点，负值更好。
仅原 dataset、N=100、1000例、greedy 与论文增强倍数匹配的行计算论文差值。
小样本/无增强仅用于检查通路，不视为论文成绩复现。论文数值保留原表精度。

| 运行/模型 | 任务 | N | 实例数 | 增强 | 本次 gap % | 论文 URS % | 论文 STL % | ΔURS pp | ΔSTL pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| code_smoke/unified_checkpoint_500 | cvrp | 100 | 2 | 1 | 1.456 | 1.810 | 1.630 | — | — |
| code_smoke/unified_checkpoint_500 | cvrptw | 100 | 2 | 1 | 6.841 | 6.130 | 4.500 | — | — |
| code_smoke/average_2 | cvrp | 100 | 2 | 1 | 16.980 | 1.810 | 1.630 | — | — |
| code_smoke/average_2 | cvrptw | 100 | 2 | 1 | 35.631 | 6.130 | 4.500 | — | — |

## 同一次运行、同一规模内的任务等权均值

不同运行分别显示；比较模型时应使用相同任务集合。

| 运行/模型 | N | 任务集合 | 本次平均 gap % | 可与论文比较数 | 平均 ΔURS pp |
|---|---:|---|---:|---:|---:|
| code_smoke/unified_checkpoint_500 | 100 | cvrp, cvrptw | 4.148 | 0 | — |
| code_smoke/average_2 | 100 | cvrp, cvrptw | 26.306 | 0 | — |

论文来源：arXiv:2509.23413v3，Table 2/3；详见 merge_lab/paper_results.md。
