# URS 单任务模型合并：定向近邻查新

检索日期：2026-11-16（Asia/Shanghai）。用途：固定 idea 的 motivation pilot 规划，不生成新 idea，不宣称穷尽文献或已确认创新性。本轮未运行实验。

## 核心判断

建议继续做 feasibility pilot；但“把单任务模型合成为多任务求解器”和“在 VRP 中使用 TIES”均已有接近工作。更准确的实验问题是：**现成 URS specialist checkpoint 能否通过一次离线参数合并，得到固定大小的单一通用模型，在一致推理预算下接近 URS generalist？** 这仍是待验证命题，不能从通用 merging 文献推出必然更好。

## 一手近邻与方法参考

| 工作 | 读到的内容 | 对当前方案的意义 |
|---|---|---|
| **MPSF-FL: Enhancing Cross-Problem Vehicle Routing via Federated Learning**，arXiv 2604.10652，2026-04-12 | 已读 HTML §4.2–4.4、Algorithm 1、§5 设置。先联合预训练 6 类 VRP，再对 10 类复杂 VRP 做多轮客户端单任务微调与 TIES 聚合；实验骨干为 MVMoE/CaDA。 | 最直接的参数 merging 近邻；不能声称首次在 VRP 采用 TIES。区别可设为现成 URS 专家、一次离线合并、固定单一部署模型，而不是反复联邦训练。 |
| **MoSES: Multi-Task Vehicle Routing Solver via Mixture of Specialized Experts under State-Decomposable MDP**，NeurIPS 2025，arXiv 2510.21453 | 已读 HTML §4.3、附录 related work；CVRP 共享骨干、单约束 LoRA 专家、逐层逐 token 动态 gating。 | 已有“复用专门求解器构建统一求解器”的动机。它属于动态专家/模块组合，不等于把 11 个现成 URS 权重离线压成一个固定参数集。其关于整模型合并 OOD 较差的说法不能当作 URS 实测结论。 |
| **Git Re-Basin: Merging Models modulo Permutation Symmetries**，ICLR 2023，arXiv 2209.04836 | 已核实摘要与全文方法目录：activation matching、weight matching、STE 寻找功能保持的神经元排列后合并。 | 若 URS 专家独立初始化，它是对齐后合并的参考。必须先为 URS 注意力/残差/超网络写正确 permutation specification 并检验重参数化前后输出不变；不能随意按层排序。无保证可直接迁移到多任务 RL 策略。 |
| **Merging Models with Fisher-Weighted Averaging**，NeurIPS 2022，arXiv 2111.09832 | 已读原论文摘要及 Fisher 合并段：每个参数按对角 Fisher 加权；作者明确限定同初始化模型，远距离参数可能失效。 | 可列为后续数据辅助 baseline，不能把它当作无共同基点时的自动补救。需要估计梯度/Fisher 的样本或预存统计，且 routing 的 score-function Fisher 定义需明确。 |
| **Dataless Knowledge Fusion by Merging Weights of Language Models (RegMean)**，ICLR 2023，arXiv 2212.09849 | 已读 HTML §3 与 Algorithm 1：线性层用各模型输入 Gram 矩阵求闭式合并，非线性层参数平均；Gram 可预先计算并共享。 | “Dataless”指合并阶段可只取预计算统计，不是统计从未使用数据。本项目只有 checkpoint 时仍需校准实例/轨迹来采激活；需单列为 calibration-assisted merge。 |
| **Linear Mode Connectivity between Multiple Models modulo Permutation Symmetries (STE-MM)**，ICML 2025 | 已核实 PMLR 摘要与官方 repo：两模型 permutation 方法扩展到多个模型可失效，提出联合寻找多模型排列的方法。 | 11 专家合并不能仅由成功的两模型对齐推断；应按 2→少量→11 的阶梯验证。实验主要是视觉分类，不构成 URS 理论保证。 |
| **Large Language Models as End-to-end Combinatorial Optimization Solvers (LLMCoSolver)**，NeurIPS 2025，arXiv 2509.16865 | 官方 README 的 “Model Merging” 指将某个训练后 LoRA 加回 base model。 | 检索误报需排除：此处 merge 是 adapter materialization，不足以证明多个任务专家的能力合并已完成。 |
| **Model Merging by Output-Space Projection**，arXiv 2605.29101，2026-05-27 | 只核实一手摘要：用校准输入及专家输出构造二次规划，另有逐层扩展。 | 可作后续 calibration-assisted 方法参考；当前不列第一轮必跑，也不根据摘要推断适配 URS 细节。 |

### 最直接近邻的关键细节

MPSF-FL §4.4 使用当前全局模型作为共同参考：`tau_k = theta_k^(t+1) - theta^t`，依次 trim、sign voting、disjoint mean，再 `theta^(t+1) = theta^t + lambda * tau`。Algorithm 1 的正式输出是 **客户端模型集合 `{theta_1^T, ..., theta_N^T}`**；全局合并模型在循环中存在并被广播。不能将其论文结果直接描述成“一次合并产生一个最终部署通用模型”的证据，也不能因为输出为 local 就否认它已经做了参数合并。

Fisher 合并每个坐标为 `sum_i lambda_i F_ij theta_ij / sum_i lambda_i F_ij`；分母近零需约定回退策略。RegMean 线性层形如 `(sum_i G_i)^(-1) sum_i G_i W_i`，并衰减 Gram 非对角元素。两者都是数学上可参考的算法，是否适合 URS 要由实际模型结构和 pilot 决定。

## 直接来源

- MPSF-FL：[摘要](https://arxiv.org/abs/2604.10652)、[全文 §4.4 与 Algorithm 1](https://arxiv.org/html/2604.10652v1)。
- MoSES：[摘要](https://arxiv.org/abs/2510.21453)、[全文](https://arxiv.org/html/2510.21453v1)、[作者代码](https://github.com/panyxy/moses_vrp)。
- Git Re-Basin：[摘要](https://arxiv.org/abs/2209.04836)、[全文](https://arxiv.org/html/2209.04836v6)。
- Fisher：[摘要](https://arxiv.org/abs/2111.09832)、[NeurIPS 原文](https://proceedings.neurips.cc/paper_files/paper/2022/file/70c26937fbf3d4600b69a129031b66ec-Paper-Conference.pdf)。
- RegMean：[摘要](https://arxiv.org/abs/2212.09849)、[全文](https://arxiv.org/html/2212.09849v4)。
- STE-MM：[PMLR](https://proceedings.mlr.press/v267/ito25a.html)、[作者代码](https://github.com/e5-a/STE-MM)。
- LLMCoSolver：[作者 README](https://github.com/Summer142857/LLMCoSolver)。
- Output-Space Projection：[摘要](https://arxiv.org/abs/2605.29101)。

## 检索覆盖与局限

执行的主要查询：

- `"model merging" "combinatorial optimization"`
- `"model merging" "vehicle routing"`
- `"model fusion" "neural" "routing" solver`
- `"task arithmetic" "routing"`
- `"neural combinatorial optimization" "merging"`
- `"routing solver" "merging"`
- `"model merging" TSP CVRP`
- `"model merging" "vehicle routing" 2025 2026`
- `"parameter merging" "routing" neural`
- `"weight merging" "VRP"`
- `"model merging" "routing problems"`
- `"model merging" "routing" "URS"`
- `"neural combinatorial optimization" "weight averaging"`
- 再以相关论文全名检索并回到 arXiv / PMLR / 官方 proceedings / 作者 repo 核实。

截至本轮，最直接的命中是 MPSF-FL，动机近邻是 MoSES；没有检出与“URS 11 个现成专家、一次离线固定参数合并、对比 URS generalist”完全同设定的已验证论文。**这只是本轮检索结果，不等于该设定不存在先行工作，也不足以确认论文创新性。** 本轮没有覆盖付费数据库或所有未索引预印本；未引用晚于 2026-11-16 的论文。

## 对最小实验的建议

1. 先审核 checkpoint 是否共享真实初始化/预训练基点；模型结构相同并不能证明权重坐标对齐。
2. 第一轮以简单平均/双模型插值作兼容性探针；有可验证共同基点才把 Task Arithmetic / TIES / DARE 标成原算法复现。
3. 无基点时先报告“现成专家合并”的结果；若另训共同起点专家，应另设实验轨道，不能混成使用现成专家零训练得到的结果。
4. URS generalist 只作为对照，不混入候选 merge 或用作虚构共同基点，否则无法归因于单任务知识合并。
5. 使用固定单一 checkpoint、多任务同一参数与一致解码预算；按任务报告目标值差距及失败率。优先证明已见任务能力保留，再测未见约束组合。
6. 接近 URS 可以支持可行性，但不能自动支持总训练成本更低；专家预训练成本与本次边际合并成本需分开。
