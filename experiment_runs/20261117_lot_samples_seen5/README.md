# LOT 与 CAT 样本量实验

使用旧 URS 多任务 base 与相同 base 微调得到的五个专家。

本轮已完成 **59 项新评测 + 9 项历史 CAT 复用**，原数据、原评测协议不变。五任务平均 gap 越低越好：旧 base **3.543209%**；LOT 官方默认 N64/alpha1.2 **3.449634 ± 0.001870%**，三次取样每次均改善全部五任务；LOT N1000/alpha1.2 **3.444764%**，为固定全量单次结果。

LOT 增加样本有效，但主要改善在 2→64，之后收益较小。CAT 论文公式 alpha0.2 从 N2 的 **3.485566%** 变为 N1000 的 **3.490645%**，没有受益；alpha0.4 从 **3.548433%** 小幅降到 **3.541014%**，接近旧 base；发布代码适配版所有样本量仍弱于 base。增加样本不是普遍有效的性能提升手段。

两者使用无标签输入估计特征统计，不进行梯度训练。论文用少量样本不等于原数据集很小：图像中多个 patch、URS 中多个节点均提供层输入特征。五任务每任务1000实例的特征采集实测约4秒，LOT合并约1秒。固定任务条件的特征不随实例数变化，已实测同seed的N2与N1000共11个超网络参数张量完全相同。

本次特征来自用户指定的原评测输入，参数比较属于同数据探索性实验。与LOT原论文使用训练集exemplars的协议差异详见PLAN.md。

- [预设方案与必要适配](PLAN.md)
- [完整结果](RESULTS.md)
- [汇总 CSV](summary.csv) · [原始逐模型逐任务结果](all_results.csv)
- [样本数量曲线](sample_count_sensitivity.png) · [LOT 系数曲线](lot_scale_sensitivity.png)
- [来源及数据哈希](manifest.json) · [固定评测计划](evaluation_plan.json)
- [独立核查](AUDIT.md) · [最终数值核查](final_verification.json)

核心实现为 `merge_lab/lot.py`，共享特征采集为 `merge_lab/cat_features.py`。官方源码快照在 `third_party/lot_merging/`。`prepare.py` 固定计划并验证输入哈希；`build_models.py` 采集特征与合并；`run_queue.py` 使用两张 GPU 调用原 URS 评测；`analyze.py` 与 `plot_results.py` 生成结果。

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
python experiment_runs/20261117_lot_samples_seen5/analyze.py --partial
```

完整执行命令见 PLAN.md。大模型、特征缓存和原始数据不纳入 Git。
