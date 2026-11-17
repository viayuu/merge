# CSP + LOT：五个 URS 微调专家

## 输入与固定对照

旧 URS 多任务 `unified_checkpoint_500.pt` 为唯一 base；五专家来自同一个 base，分别微调 CVRP、CVRPB、CVRPTW、OCVRP、OCVRPTW 30 epoch。沿用原五任务 N100、各1000实例、原参考解、greedy、8倍增强、batch125；CVRPB POMO80，其余 POMO100。既有4090续训不参与本轮，也不暂停。

输入模型、数据、参考解与前轮 LOT 的哈希核验一致。使用前轮 N64/seed3407、3408、3409 和 N1000/seed3407 完整专家特征缓存，每份缓存哈希写入 manifest。N1000 是全量固定集合，不伪造独立取样重复。

## 官方方法与适配

文献：[ICML 2026 / PMLR 306](https://proceedings.mlr.press/v306/sun26r.html)。源码：[SunWenJu123/model-merging](https://github.com/SunWenJu123/model-merging)，提交 `ab87731e405b9eeca1f822685ede0276d756907c`，快照在 `third_party/csp_merging/`。

1. 先通过 LOT 得到未缩放的合并增量 T。
2. 对每个二维 base 权重作 FP32 `torch.linalg.svd(full_matrices=False)`，取前 r 个左右奇异向量。
3. 保留 `(I-U_r U_r.T) T (I-V_r V_r.T)`，按照官方分解计算，不构造巨大单位/投影矩阵，不进行额外重缩放。
4. 最终模型为 `base + alpha * projected_T`。所有一维参数保留 LOT 的融合增量。

官方 CSP 入口调用 `lot_merging`，现有适配调用内存更省的 `lot_mergingv2`；二者线性层伪逆与归一化加权公式相同。CPU原函数对照差为0。v1的首专家缺失特征边界在这五个任务中不触发，builder检查五任务inactive完全一致。保留前轮 URS 模块类型识别、完整 InstanceNorm 特征、固定任务条件计一次等必要适配。

CSP遍历全部二维参数，包括名字含bias的矩阵。`r=min(k,min(shape))` 完全沿用官方行为：k=5时40个小矩阵的二维增量理论上被清空；k≥13还会清空13维条件/超网络矩阵的增量。实际FP32微小舍入残差保留，不偷偷截断rank或恢复更新。每层rank、被保护的能量和增量保留比例均落盘。

官方0<k<1分支按奇异值**平方和**累计比例选秩；论文附录C.4文字写奇异值之和，本轮明确沿发布代码。附录没有给CSP+LOT的p默认值，故以下能量参数属于URS补充。无CSP对照显式绕过投影；不能直接把官方k设为0，因为该分支仍会取一个方向。

## 评测前固定的100条计划

| 设置 | 特征样本/任务 | alpha | 取样重复 | 秩或能量参数 | 新评测数 |
|---|---:|---:|---:|---|---:|
| 旧URS base | — | — | 1 | — | 1 |
| 官方入口对应的纯LOT对照 | 64 | 1.4 | 3 | 无投影 | 3 |
| CSP+LOT官方秩扫描 | 64 | 1.4 | 3 | r=1,2,…,20 | 60 |
| 能量秩补充 | 64 | 1.4 | 3 | p=0.1,0.5,0.9 | 9 |
| 上一轮最佳LOT配置复跑 | 1000 | 1.2 | 1 | 无投影 | 1 |
| CSP+LOT全量样本补充 | 1000 | 1.2 | 1 | r=1,2,…,20 | 20 |
| 能量秩补充 | 1000 | 1.2 | 1 | p=0.1,0.5,0.9 | 3 |

另复用前轮纯LOT N64/alpha1.2三次结果，供上下文比较。合计97项新评测+3项历史复用，构建96个新模型（不含已有base），包含92个CSP+LOT模型。官方入口初始k=5随后被1…20循环覆盖，因此r5标为入口预设而非唯一官方运行配置。

主问题是CSP能否改善LOT，而不只是是否胜过旧base。每个CSP配置同时与旧base及**同样本量、同alpha、同seed的纯LOT**比较。64/1.4与1000/1.2同时改变了样本和系数，不能将两组差异单独归因于样本。保留所有预设秩和差结果，按完整配置均值比较，不挑seed。

按用户要求不另建数据集、不另拆分。LOT特征只读取原输入，不读解/目标；CSP本身不需要额外样本。最终与相同原数据上的URS结果比较，超参选择属于同数据探索性结果。论文关于base核心子空间近最优的假设未在URS上获得证明，性能由本轮实验判断。

## 执行与核查

沿用 `conda activate easynco`，Slurm 1465/gpu03 两张 RTX3090，tmux持久运行。base SVD先单进程缓存，随后两卡并行构建与评测。构建时检查所有参数有限且原URS模型严格加载；smoke单独使用2实例，不能计入正式统计。

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
python experiment_runs/20261117_csp_lot_seen5/prepare.py
bash experiment_runs/20261117_csp_lot_seen5/launch.sh smoke
bash experiment_runs/20261117_csp_lot_seen5/launch.sh
```

运行中可用 `analyze.py --partial`；完成后自动生成原始CSV、配置汇总、结果报告和rank/energy曲线。最终独立核查模型来源、原协议、计划完整性、均值/std与逐任务改善。
