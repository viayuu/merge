# LOT merging 与 CAT 样本量实验

## 固定输入与对照

- Base：URS 原始 `pretrained/unified_checkpoint_500.pt`，不是续训后的多任务模型。
- 专家：从该 base 微调 30 epoch 的 CVRP、CVRPB、CVRPTW、OCVRP、OCVRPTW 五个模型。
- 输入模型、原数据、参考解的 SHA256 记录在 `manifest.json`；`prepare.py` 检查与前轮 CAT 的记录一致。
- 每个合并模型均使用原 URS 五任务、规模 100、每任务原始 1000 个实例、greedy、8 倍增强评测。保留原参考解和任务特定 POMO 设置。
- 使用已有 Slurm 1465 的两张 RTX 3090；不改变 4090 上的多任务续训。

## LOT 实现与必要适配

官方仓库：[SunWenJu123/model-merging](https://github.com/SunWenJu123/model-merging)，固定提交 `ab87731e405b9eeca1f822685ede0276d756907c`。原文件存于 `third_party/lot_merging/`。论文：https://arxiv.org/html/2505.23859v2 。

- 保留 `lot_mergingv2` 的闭式计算：线性层输入 Gram 为 C_k=X_k^T X_k，融合增量转置为 `pinv(sum C_k) @ sum(C_k @ delta_k.T)`。
- 使用原实现 FP32、PyTorch 默认伪逆；不添加 ridge、不更改秩阈值、不补零空间、不增加训练迭代。
- 完整专家提供特征；最终模型为 `base + alpha * fused_delta`。官方 ViT-B-32 默认是每任务 64 个样本、alpha=1.2，原样作为本实验主配置。
- URS 使用模块类型识别 Linear、InstanceNorm1d 与 bias，替代 CLIP 层名规则。归一化 scale 按标准化输入平方和加权；bias 和未激活层保留专家增量的均值。
- 执行真实 `decoder.assign(13维任务条件)` 与 `model.pre_forward`，采集持久参数的输入。超网络生成的 decoder 权重不是单独的 checkpoint 参数；本实验未另行增加生成权重的损失或 rollout 特征。
- 每任务的条件固定，只统计一次；增加实例数会增加节点/归一化特征，但不会增加不同的条件。条件 Gram 秩不足由官方伪逆处理。
- 分批 128 个实例累计充分统计，避免大样本显存开销。与未分批采集的 CPU 对照验证：CVRPTW 4 个实例一次处理与 2+2 分批，Gram 相对误差 0，归一化统计最大相对误差约 1.79e-7，计数相同。
- 数学对照：仅将官方 `lot_mergingv2` 的 CUDA 调用改为 CPU，对同一合成输入，线性层、bias、未激活层差为 0，归一化 scale 最大绝对差约 1.19e-7。

## 预先固定的比较

每任务样本量：**2、16、64、256、1000**。

- 2/16/64/256 分别用 seed 3407、3408、3409；每个 seed 对每个任务生成一个排列，取相同排列的前 N 个，形成嵌套子集。
- 1000 使用全部原实例、原顺序，仅做一次；不能把全量的不同排列当成独立抽样重复，标准差标为 NA。
- LOT：所有样本量固定 alpha=1.2；在默认 N=64 上另做 alpha=0.6、0.8、1.0、1.5、2.0 的 URS 探索性扫描。该扩展扫描不是声称复现论文的原始扫描网格。
- CAT：固定 paper-full/c4/alpha=0.2、paper-full/c4/alpha=0.4、published-code/c4/alpha=0.4 三种配置。沿用前轮实现、投影参数和差异解释，仅改变实例数量。
- N=2 的 CAT 三配置各三次结果直接复用前轮；LOT N=2 使用前轮已有特征缓存。其余重建特征和模型。
- 共 68 条五任务评测记录，其中 9 条复用历史 CAT，59 条新评测（58 个新合并模型加旧 base 的一次重新评测）。

LOT 的 alpha 乘以闭式融合增量；CAT 的 alpha 乘以裁剪后增量之和，两者不能仅按相同 alpha 数值直接解释。

## 数据用途与结论边界

按用户要求，不另建数据集或划分。特征采集只读取原实例输入，不读取参考解/目标值；最终在同一原始 1000 个实例上评测。这是使用评测输入进行无标签适配的设置，与 LOT 论文采用训练集 exemplars、测试集评测的协议有区别。参数扫描最优值属于同数据上的探索性结果，不作为独立测试集泛化证据。

两篇论文的少样本设置不是因为数据集本来很小。CAT 一张图像有多个 patch 特征；LOT 的样本量消融包含 4–64，并推荐 64。URS 的每个 100 客户实例也提供 101 个节点特征。样本量比较需要固定其余设置，实际观察收益、方差和耗时，不能预设更多一定更好。

## 执行与产物

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
python experiment_runs/20261117_lot_samples_seen5/prepare.py
bash experiment_runs/20261117_lot_samples_seen5/launch.sh smoke
bash experiment_runs/20261117_lot_samples_seen5/launch.sh
```

长任务放入 tmux。队列保存逐模型状态、原始 URS 输出与日志，完成后生成 `RESULTS.md`、CSV、JSON 与 PNG/PDF 曲线。运行时可用 `analyze.py --partial` 汇总已有结果。耗时区分特征采集、合并和评测；共享特征与融合向量的复用不重复算新增耗时。
