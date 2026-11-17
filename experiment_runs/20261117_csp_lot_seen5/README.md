# CSP + LOT：URS 五任务实验

使用旧 URS 多任务 base 和五个共同 base 微调专家，先做 LOT，再按官方实现做 CSP 双侧投影。

**已完成100/100项记录（97新评测+3历史复用）。本轮92个CSP模型的五任务平均gap全部弱于匹配纯LOT，也全部弱于旧base。当前结果不支持在URS上直接叠加这套CSP投影。**

gap单位为%，越低越好；旧base为3.543209%。

| 设置 | 纯LOT | 本轮最佳CSP+LOT | 较纯LOT（百分点） |
|---|---:|---:|---:|
| N64，alpha1.4，三次取样 | 3.494177 ± 0.003263 | r16：3.641925 ± 0.007905 | +0.147749 |
| N1000，alpha1.2，全量一次 | 3.444764 | r16：3.549785 | +0.105022 |

官方入口预设r5、N64、alpha1.4的结果为4.687095 ± 0.012977%。完整扫描了r1–20，并补测p=0.1/0.5/0.9的逐层平方能量选秩。两条设置的能量方案均未改善纯LOT。

独立核验未发现增量定义、缩放、转置或模型来源错误，原始LOT缓存能逐位重建上一轮模型。CSP对URS的小矩阵和条件层有强约束：r5理论上消除40个二维矩阵更新；r≥13消除85个。这个结构现象已证实，但它是否造成主要退化仍需另外的消融验证；本轮未为改善结果而改投影或跳过层。

两张RTX3090正式评测用时约39.1分钟，累计评测GPU时间约1.288小时。原数据与原协议不变，未新增划分；这些参数比较属于同数据探索性结果。

- [预设方案与必要适配](PLAN.md)
- [完整结果与逐任务对比](RESULTS.md)
- [配置汇总 CSV](summary.csv) · [逐模型逐任务 CSV](all_results.csv)
- [固定秩曲线](rank_sensitivity.png) · [逐层能量比例曲线](energy_sensitivity.png)
- [输入来源与哈希](manifest.json) · [固定计划](evaluation_plan.json)
- [代码核查](AUDIT.md)
- [最终结果核验](final_verification.json)

核心实现：`merge_lab/csp.py`；LOT沿用 `merge_lab/lot.py`；参考源码在 `third_party/csp_merging/`。所有CSP结果同时对比旧base和同样本、同系数、同seed的纯LOT，避免把改变系数的影响误认为CSP收益。

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
python experiment_runs/20261117_csp_lot_seen5/analyze.py --partial
```

长任务由tmux与Slurm运行，两张RTX3090。具体启动命令见PLAN.md。模型、SVD/LOT缓存均由现有Git忽略规则排除。
