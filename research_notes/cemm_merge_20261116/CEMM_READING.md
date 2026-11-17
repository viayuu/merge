# CEMM：论文、附录与代码核对

2026-11-16。依据用户提供的 9 页正文、10 页技术附录和代码 ZIP。稿件首页标注 Anonymous submission；没有核实作者、正式会议或发表年份，不沿用历史对话中“可能 AAAI 2027”等猜测。

原件：[正文](/public/home/shiys/merge/CEMM/40088_Large_Language_Model_Emp.pdf)、[附录](</public/home/shiys/merge/CEMM/40088_Large_Language_Model_Emp_Technical supplement.pdf>)。代码解压到 `/public/home/shiys/merge/CEMM/CEMM-main/`。本次仅阅读与静态审查，未运行它的 LLM、训练或求解流程。

## 1. 论文解决什么问题

论文希望把多个 VRP 专家的能力整合到通用模型。它认为固定合并规则未必适合不断变化的模型；只按平均成绩保留模型，又可能丢掉在某项任务特别优秀的专家。

因此它同时维护两类对象：

- **算法种群**：不同的合并程序，用 LLM 产生新规则。
- **模型种群**：一个通用模型和每个任务的专家角色；评测新候选后，分别更新这些角色。

贡献在于如何生成、评价和共同演化这些对象。它不是只有一个固定的“CEMM 参数平均公式”。

## 2. Base 和专家具体从哪里来

附录 A 的 RF 实验设置：

| 项目 | 原文设置 |
|---|---|
| 共同起点 | RouteFinder 第 270 epoch checkpoint |
| 专家任务 | CVRP、OVRP、VRPB、VRPL、VRPTW |
| 专家训练 | 每任务在同一起点上微调 30 epoch |
| 每 epoch 数据量 | 100K 实例 |
| 优化器/学习率 | Adam；3×10⁻⁵；最后 5 epoch 乘 0.1 |
| batch | N=50 为 256；N=100 为 128 |

正文第 5 页说明 POMO-MTL 遵循相同的预训练、微调协议。因此 POMO 也需要自己的公共模型及分任务专家，不能拿单个 POMO 模型直接完成专家合并。MTL 对照从起点在混合任务数据上继续训练。

原文没有给出“270＋30 通过消融证明最优”的证据。它是采用的配置；不能把助手关于学习率节点的解释升级成作者已经验证的选择依据。

**CEMM 合并阶段避免重新训练完整求解器，不等于它从头到尾没用多任务数据、没训练专家或没做优化。** 这与当前 URS 发布专家从随机起点长期训练的设置不同。

## 3. 一轮到底发生什么

设当前模型是通用模型 G 和五个专家 E₁…E₅。一个算法个体写成 h=(d,c)：

- d 是自然语言描述，例如“对五个专家逐坐标取中位数”。
- c 是执行该规则的 Python 函数，接收一个参数字典和若干专家参数字典，返回新模型参数。

流程是：参考已有程序生成候选程序 → 候选程序作用于模型快照 → 得到候选模型 → 在五项任务评估 → 更新算法种群、通用模型和各任务专家 → 继续下一批。

模型更新的两种标准不同：平均 gap 更低，才替换 G；任务 t 的 gap 更低，才替换 Eₜ。因此一个候选即使平均分一般，也可能保留为某任务专家。后期“专家”是角色定义，并不一定仍是最初那个单任务微调 checkpoint。

代码使用每个算子批次固定输入快照。该批五个候选使用相同模型，结束后再更新；不是每生成一个候选就立刻改变下一个候选的输入。

证据：

- [执行合并函数](/public/home/shiys/merge/CEMM/CEMM-main/cemm/src/cemm/problems/optimization/model_merge/run.py:771)：第 0 个模型作为 `base_model`，其余作为专家，调用 `merge_function`。
- [任务专家更新](/public/home/shiys/merge/CEMM/CEMM-main/cemm/src/cemm/problems/optimization/model_merge/run.py:884)、[通用模型更新](/public/home/shiys/merge/CEMM/CEMM-main/cemm/src/cemm/problems/optimization/model_merge/run.py:942)。
- [批次快照](/public/home/shiys/merge/CEMM/CEMM-main/cemm/src/cemm/problems/optimization/model_merge/run.py:482)、[写回当前 generalist](/public/home/shiys/merge/CEMM/CEMM-main/cemm/src/cemm/problems/optimization/model_merge/run.py:517)。

**每轮函数里的 `base` 会变成演化中的 G，不永远等于历史 epoch270 的 θ₀。** “专家最初共享哪一个起点”和“当前合并规则参照哪一个参数字典”是不同问题。最终模型也可能由后来已淘汰的程序产生，不能把最终模型与最终算法种群当作同一个产物。

## 4. LLM 实际能看到什么

| 操作 | 代码中的参考材料 |
|---|---|
| E1，对应 `d1` | 当前全部算法的描述；只有第一个附代码 |
| E2 | 两个被抽到的父算法的描述及代码 |
| M1/M2 | 一个被抽到的父算法的描述及代码 |
| 运行错误修复 | 失败代码及异常信息 |

正常生成 prompt 没有拼接全部历史成绩表、各任务 gap 或模型权重。评测通过外部选择影响“哪些程序能进入 prompt”，属于间接反馈；不能说 LLM 每次会根据真实性能曲线分析失败原因。

代码位置：[prompt 组装](/public/home/shiys/merge/CEMM/CEMM-main/cemm/src/cemm/methods/eoh/eoh_evolution.py:80)、[单条请求消息](/public/home/shiys/merge/CEMM/CEMM-main/cemm/src/cemm/llm/api_general.py:17)。

父算法不是均匀随机选：[prob_rank.py](/public/home/shiys/merge/CEMM/CEMM-main/cemm/src/cemm/methods/selection/prob_rank.py:2) 按数组位置 r 赋权 1/(r+1+N)，再有放回抽样。因此 E2 可能两次抽中同一个算法；该数组顺序也未必等于严格平均 gap 排名。

## 5. “互补”评价什么，论文与代码有何区别

已选算法集合 S 在任务 t 上的最好 gap 为 bₜ。正文第 4 页 Eq.1 衡量候选 h 带来的相对改善：

\[
\Delta(h\mid S)=\frac1M\sum_t\max\left(\frac{b_t-g_h^t}{b_t+\epsilon},0\right).
\]

先保留平均成绩最好的程序，再保留能补足其任务短板的程序。b 向量的每一项可能来自不同候选，它是集合覆盖能力，**不是某个真实单模型已经达到的成绩**。这也不是逐实例路线互补性的直接测量。

[补充代码](/public/home/shiys/merge/CEMM/CEMM-main/cemm/src/cemm/methods/management/pop_complementary.py:88) 实际使用：

\[
\Delta_{\rm code}(h\mid S)=\sum_t\max(b_t-g_h^t,0).
\]

缺少 1/M 不改变候选排序；缺少每个任务的分母可能改变排序。因此论文的归一化规则与提供的实现不能写成完全一致。代码还按平均 objective 四舍五入五位去重，可能将平均分相同而任务专长不同的候选合并掉。

**迁移 URS 时的一个额外问题（本次代数推导）：** PCTSP 的原 gap 可能为负。如果 b=-0.08、候选更好为 -0.10，则正改善 0.02 除以负分母后被截为零；更差的 -0.06 反而可能得到正分。因此不能直接照搬这个相对改善公式到包含负 gap 的任务。需要处理选择分数的尺度与符号；这不要求改变你指定的数据、参考结果或最终报告 gap。原始改善求和版本没有这个符号反转，但仍存在任务尺度差异。

## 6. 附录两个合并程序

附录 D 明确将二者称为演化过程中生成的例子。不能声称所有最终 CEMM 模型都统一由它们中的某一个产生。

**算法 1：逐坐标中位数。**

\[
\theta_m=\theta_0+\operatorname{median}_i(\theta_i-\theta_0)
=\operatorname{median}_i(\theta_i).
\]

后一个等式是平移不改变顺序的代数结论。因此计算结果不需要历史 base 文件。对某个坐标，三个专家为 0.2、0.3、2.0，均值为 0.833，中位数为 0.3；它减少极端数值的影响，但无法判断极端值究竟是噪声还是有用的任务知识。

附录代码用 `torch.median`。偶数个专家时它取较小的中间值；两个专家时是逐坐标较小值，**不是二者平均**。迁移时必须写清这一约定。无需显式 base 不代表不同初始化的参数天然可比较。

**算法 2：参数张量统计 → 小网络 → 专家权重。**

对张量 k 计算 Δᵢᵏ=θᵢᵏ−θ₀ᵏ，再取每个专家的张量均值，得到 M 维输入。全模型共用的 `WeightNet(M,16,M)` 经 ReLU 和 Softmax 输出这个张量的 M 个专家系数。它是每张量一组权重，不是每个坐标一组，也不是每个输入 VRP 实例一组。

内部 Adam 优化学习率 0.001，共 100 epoch，目标为：

\[
\min_\phi -\sum_k\left\|\sum_i w_i^{(k)}(\phi)\Delta_i^{(k)}\right\|_F.
\]

训练的是辅助权重网络；没有用路线数据训练它，但确实发生了梯度优化。其内部目标鼓励较大的合并增量范数，不直接优化路线成本、可行性或任务间一致性；外层候选评价才用 VRP gap。把它解释成“通过任务表现学出了真正重要的专家知识”超出了证据。

## 7. 哪些结果支持单模型合并

正文表 2 的五任务平均 gap，越低越好：

| RF-TE 系列 | N=50 | N=100 |
|---|---:|---:|
| 各任务对应专家 | 2.018% | 2.742% |
| RF-TE 多任务对照 | 2.108% | 2.852% |
| CEMM 单模型 | 2.034% | 2.813% |
| CEMM+，3 模型 | 1.701% | 2.471% |

单模型比 MTL 改善约 0.074 / 0.039 个百分点，是该设置下的正面证据；没有全面超过任务专家。三模型增强版的更大提升不能全部归因于单模型参数合并。上述五任务 gap 与 URS 十一任务平均值也不能跨数据、跨任务直接比较大小。

CEMM+ 从互补模型池选 k=3 个模型，各自推理再选较好的可行解。默认每个模型 ×8 augmentation，因此总 rollout 预算高于单模型。附录 E.4 另设专家 ×24 对照，N50 专家均值 1.826%、CEMM+ 1.701%；这是预算匹配的集成比较，仍不是固定单模型结果。

原 [GENOME 论文 §3.3](https://aclanthology.org/2026.acl-long.2044.pdf) 中 GENOME+ 同时加入 succession（历史经验继承）和 ensemble。CEMM 明确强调其比较中的“+”为集成推理；仅凭这个符号，不能断言它完整复现或删除了 GENOME 原文的其他增强机制。

## 8. 跨 backbone 到底迁移了什么

附录 E.2 将 RF 上发现的算法 1 原样用到 POMO 的专家集合。迁移的是 Python 合并规则，不是把 RF 权重加到 POMO 权重中，也不省掉 POMO 专家的准备。

| POMO 相关实验 | 平均 gap |
|---|---:|
| Task Arithmetic | 2.245% |
| DARE-TIES | 2.242% |
| PCB | 2.291% |
| WUDI | 2.717% |
| RF 规则直接迁移（算法 1） | 2.219% |
| POMO-MTL 对照，正文图 2 | 2.172% |
| 完整 CEMM，正文图 2 | 2.117% |

2.219% 优于所列静态合并方法，但没有优于 POMO-MTL，也不等于完整 CEMM 的 2.117%。

## 9. 搜索成本与补充代码状态

正文配置为 GPT-4o、算法种群 5、30 轮、每轮 20 个新程序，RTX3090。搜索验证样本：N50 每任务 1000，N100 每任务 128；测试每任务 1000。600 是主循环新候选数，不能当所有评估调用的上限：还包含幸存程序重评、修复和重试。搜索成本包括 LLM、合并、求解评价及选择，不只是形成最终 checkpoint 的一次矩阵运算。

补充包提供了一个实际 checkpoint：`RF-TE/checkpoint/CEMM.ckpt`，约 26.8 MB。**历史对话中“ZIP 没有 checkpoint”的说法不正确。** 但它没有包含完整的起始 base、五个专家和搜索记录，不能开箱即用复现全流程。

静态核对发现的具体缺口：

| 位置 | 问题 |
|---|---|
| [eoh_interface_EC.py:613](/public/home/shiys/merge/CEMM/CEMM-main/cemm/src/cemm/methods/eoh/eoh_interface_EC.py:613) | 前方 `if self.debug:` 空块，AST 解析为 IndentationError |
| [runEoH.py:3](/public/home/shiys/merge/CEMM/CEMM-main/examples/model_merge/runEoH.py:3) | 导入 `eoh.src.eoh`，而包为 `cemm/src/cemm` |
| [run.py:33](/public/home/shiys/merge/CEMM/CEMM-main/cemm/src/cemm/problems/optimization/model_merge/run.py:33) | 用于构建评测模型的 RF checkpoint 路径和初始 checkpoint_paths 列表为空 |
| [run.py:43](/public/home/shiys/merge/CEMM/CEMM-main/cemm/src/cemm/problems/optimization/model_merge/run.py:43) | 预期的 evaluate_model_merge 目录未包含 |
| [getParas.py:38](/public/home/shiys/merge/CEMM/CEMM-main/cemm/src/cemm/utils/getParas.py:38) | 配置指向未提供的 seeds.json |
| [run.py:61](/public/home/shiys/merge/CEMM/CEMM-main/cemm/src/cemm/problems/optimization/model_merge/run.py:61) | 初始专家和通用模型成绩硬编码，换模型需要重新取得真实值 |
| [RF-TE/test.py:146](/public/home/shiys/merge/CEMM/CEMM-main/RF-TE/test.py:146) | CUDA 路径写死 cuda:1 |

还缺少外部 RouteFinder 相关模块。`use_encoder_decoder_mode=True` 虽然出现在问题类，入口未把它传给默认 False 的 InterfaceEC；不能仅凭存在分 encoder/decoder 的函数就说默认实验已经启用。

这些发现只说明提供的补充版本有复现缺口，不说明论文结果不真实。本次没有修改这些问题或启动复现。

## 10. 对当前工作的意义

CEMM 已是“把专家合成为一个多任务 VRP 求解器”的直接先例。因此值得继承的是明确的比较对象与能力保留问题，不能把这个宽泛目标重新包装成创新。

你的 URS 设置首先面临专家来源与兼容性。CEMM 的同源短期微调成功并不能直接回答它；CEMM 的程序空间也可能包含细粒度规则，不能用“CEMM 完全不考虑参数选择”来制造区别。后续真正有说服力的区别，需要落实到可测的损伤机制、具体合并方法，以及固定单模型下的结果。
