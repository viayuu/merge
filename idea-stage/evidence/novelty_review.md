**Verdict: PROCEED — 通过进入 motivation pilot 规划的早期新颖性检查；不是论文接收判断。**

审查日期：2026-11-16。范围严格限定为：一次离线合并 11 个现成 URS specialist checkpoint，得到一个固定大小、跨任务使用相同参数的 solver，在匹配解码预算下接近已发布 URS generalist，不进行联合重训。当前没有 pilot 结果，本审查没有运行实验。

审查身份：canonical agent id 为 `/root/novelty_review`；实际 session UUID 未提供，记为 `unknown`；可核实的具体部署模型标识和 reasoning effort 均为 `unknown`，不将提示文本中的模型名称当作运行身份凭证。本审查来自同一 Codex 协作体系，应按 **same-family / provisional review** 使用，不能宣称跨模型独立验证。

**可供审稿人核验的一句话差异：** 本假设检验既有 URS 全参数单任务专家能否在不追加联合训练、不保留动态专家组合的情况下，一次离线压成一个固定参数集，并在原 11 项任务和相同解码预算下接近 URS generalist。

| 最接近的工作 | 已占据的内容 | 与固定假设的实际差异 |
|---|---|---|
| MPSF-FL，*Enhancing Cross-Problem Vehicle Routing via Federated Learning* | 多任务预训练后，反复进行客户端单任务微调与 TIES 参数聚合。它已经在 VRP 中执行参数合并。 | 本假设复用已经存在的 11 个 URS 专家，一次离线产生固定模型；不靠继续训练来形成兼容的客户端参数。MPSF-FL 的 Algorithm 1 返回客户端模型集合，表 1 标注 client-side performance；不能据此认定它已经证明本假设。[原文 §4.2–4.4、Algorithm 1、表 1](https://arxiv.org/html/2604.10652v1) |
| MoSES，*Multi-Task Vehicle Routing Solver via Mixture of Specialized Experts under State-Decomposable MDP* | 已明确提出复用专门求解器构建统一 VRP solver；使用共享 CVRP 骨干、LoRA 专家、逐层逐 token 自适应组合，还训练残差 LoRA。 | 它保留专家模块并使用输入相关组合；本假设最终只部署一套离线确定的 URS 权重。MoSES 的 latent-space 论证不证明静态权重平均能够保留能力，其引用的 OOD 劣势也不是对当前 URS 专家的直接实验。[原文 §4.3、公式 4](https://arxiv.org/html/2510.21453v1) |
| Model soups / Task Arithmetic / TIES / DARE | 固定参数量、无额外联合训练的专家融合在一般机器学习中已有成熟方法。 | 当前差异在 URS 现成 checkpoint 与 routing 策略上的可验证经验结果，不在“参数平均”“任务向量”“TIES”本身。具体方法前提见已读的本地经典文献证据。 |

MPSF-FL 的共同参考点是每轮全局模型，客户端再次从它出发训练。因此，“省掉全部训练循环还能保留 11 个现有专家的能力”是实质的待检验约束，不能只把原论文的联邦通信移到本地便视为已经完成。另一方面，不能借其最终输出为客户端模型而淡化它已经使用 TIES 的事实。[MPSF-FL 原文](https://arxiv.org/html/2604.10652v1)

**为什么给 PROCEED：** 当前读到的具体论文没有包含这一完整结果；两篇近邻的训练与部署机制存在可核验差异。仅有邻近文献、没有实验结果、或者采用经典合并算法，都不足以判定 ABANDON。这里通过的是对一个尚未被当前证据回答的经验命题开展小规模验证的门槛。若最终只有机械套用现有算法，仍不能自动宣称新方法贡献；那属于后续证据与论文定位判断，不应在规划阶段把固定假设扼杀。

**现在可安全写：** “我们计划检验：现有 URS 单任务专家是否能经一次离线参数合并，在不追加联合训练的条件下形成固定大小的统一 solver，并在匹配解码预算时接近 URS generalist。” 可写“在已核查文献中，未发现完全相同设定的已验证结果”，不能写“首次”或“文献中不存在”。

**现在不能写：** 已实现接近或超越 URS、首次在 VRP 中合并模型或使用 TIES、理论保证保留所有任务能力、总训练成本更低、天然无需任何数据、未见任务泛化更好。即便后续 11 项 seen 任务成功，也只支持预先定义协议下的能力保留；不能推出 99 项 unseen 成功。复用专家的边际合并成本与专家历史训练成本必须区分。

**固定范围内，让结果足以支撑这条差异的证据：**

1. 先确认实际 checkpoint 架构与训练谱系。共同架构不等于共同初始化；底座不明时不可将 URS generalist 作为虚构共同起点，再把其多任务能力归因于专家合并。该问题是可行性与归因风险，不是已被先行论文解决的证明。
2. 提前固定“接近”的阈值和跨任务统计方式，报告每项任务的质量与合法性；宏平均不能掩盖个别任务崩溃。阈值应在看 test 结果前确定，本审查不事后编造数值。
3. 确认同一 checkpoint 用于全部 11 项任务，并对齐实例、解码候选数量、增强与后处理等预算。若需要调合并超参，只使用独立验证集，记录所有额外数据与计算。
4. 先用规划中的小规模兼容性 pilot 判断是否值得进入 11 任务验证。失败只说明被测试的方法及其参数坐标条件未能完成假设，不支持“model merging 在 routing 中不可能”。本轮不新增研究方向或运行任何试验。

已读本地证据：`/public/home/shiys/merge/idea-stage/evidence/co_prior_art.md` 与 `/public/home/shiys/merge/idea-stage/evidence/classical_merge_literature.md`。独立打开并核对了 MPSF-FL §4.2–4.4、Algorithm 1、表 1 标题，以及 MoSES §4.3 与公式 4。经典算法细节采用已提供的文献证据，本次未重新逐篇通读其原文。

补充定向检索覆盖了 `"URS" "model merging" routing`、`"Universal Routing Solver" "merging"`、`"one-shot" "merging" "vehicle routing"`、`"model merging" "routing" "specialists"`、`"model merging" "combinatorial optimization"`、`"parameter merging" "vehicle routing"`、`"weight merging" "routing solver"`、`"URS" "single-task" "merge" routing`。没有新增命中可直接覆盖固定结果的论文；通用 LLM routing、软件模型合并及将单个 LoRA 加回底座不构成该结果的先例。检索有索引与覆盖限制，不声称穷尽所有公开或未索引工作。当前判决不依赖“查无结果即证明不存在”，而依赖已识别近邻与固定假设之间的明确差异。
