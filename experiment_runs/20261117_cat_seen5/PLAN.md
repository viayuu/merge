# 五个 URS 专家的 CAT Merging 实验

固定输入：原 URS `unified_checkpoint_500.pt`，以及 CVRP、CVRPB、CVRPTW、OCVRP、OCVRPTW 的已完成 epoch-30 微调专家。继续训练中的多任务模型不参与本轮。计算使用 allocation 1465 / gpu03 的两张 RTX 3090；不干扰两张 4090 的续训。

## 来源和两个版本

论文：ICML 2025，https://proceedings.mlr.press/v267/sun25i.html ，arXiv 2505.06977v2。
官方仓库：https://github.com/SunWenJu123/model-merging ，固定 commit `ab87731e405b9eeca1f822685ede0276d756907c`。上游文件保存在 `third_party/cat_merging/`，算法适配在 `merge_lab/cat.py`。

公开代码与论文存在可验证差异，因此分别报告，不能把两种结果混称为原样复现：

| 项目 | 发布代码行为（upstream） | 按论文公式实现（paper） |
|---|---|---|
| 采集特征的模型 | base + 0.4 delta | 完整专家 base + delta |
| 线性方向 | eigh 升序的前 c 列；无正特征值则跳过 | 最大的 c 个特征方向，Eq.11 |
| 连续投影后 | 每次恢复增量原范数 | 按公式直接投影，不恢复范数 |
| norm scale / bias | 入口覆盖为关闭 | 开启 Eq.14 / Eq.17 的跨任务 mask |
| bias 数量 | 顶部写25%，但入口实际关闭 | 每张量删除 c 个坐标，采用论文定义 |

官方顶部 ViT-B/32 设置：alpha=.4、样本2/任务、重复3次、linear c4/lambda5、norm c4/lambda10；入口将norm和bias关闭，并将测试列表清空。URS版本接入原评测器，不复制空测试列表。上游bias分支索引与论文不同，本轮upstream路径本来关闭此分支，不使用有歧义的bias实现。paper版本用论文公式，不宣称其超参数是URS上的最优配置。

## 固定配置（评测前确定）

- 发布代码适配：alpha=.4，c4/lambda5，norm/bias关闭，3次取样。
- 论文公式完整CAT：linear c4/lambda5、norm c4/lambda10、bias c4，alpha=.4作为预设参考。
- 完整CAT的alpha单因素扫描：.1,.2,...,.9，与论文敏感性图范围相同；每个点3次取样。
- 完整CAT的c单因素补充：c2和c3，alpha=.4，每个点3次取样；linear/norm/bias一同调整c。
- 论文公式仅线性：c4/lambda5、alpha=.4，3次取样，用于分离norm/bias的影响。
- 共13种CAT配置×3次=39个模型，另在同卡复测旧base和未裁剪Task Arithmetic alpha=.4，共41次完整五任务评测。此前平均/TIES结果作为明确标注的历史参考。

取样seed固定3407/3408/3409。每个seed按任务顺序各从原1000例中无放回取2例（索引写入构建记录），不同算法/alpha使用相同索引。只读实例，不读取参考解或gap来构造CAT。原始代码用训练loader，本项目按用户要求直接使用已有评测输入，不新建split；因此这是使用评测输入的无标签适配，必须明确记录。扫描最优同样是同数据探索性最优，不作独立泛化估计。重复只反映少量样本选择的变化，不是重新训练专家。

## URS 必要适配

1. 图像前向替换为 `decoder.assign + model.pre_forward`。所有持久可训练参数均在此使用；随后POMO解码使用已生成权重，不产生新的参数层输入，因此不需完整rollout采特征。两个实例不增强，节点作为特征行。
2. 模块各注册一次hook，直接累计X^T X；等价于保存所有输入行再求Gram。归一化替换为URS的InstanceNorm1d，恢复仿射前归一化特征，按通道统计平方和。paper按sum，未误用mean。
3. 通过模块类型区分linear/norm/bias。upstream线性分支仍保留官方`'bias' in name`排除规则，因此URS的adaptation_bias_module内权重也跳过；paper用模块类型覆盖这些线性层。未激活relation模块不投影，其零增量保持不变。
4. URS的4个16384×13超网络矩阵，使用增量列空间QR后计算小冲突矩阵。实际谱核查：两种特征模型×4层×5保护任务的40个矩阵均有4个显著正方向和4个显著负方向，当前c≤4只选择这些非零方向，因此等价于大矩阵的极端特征子空间。代码逐次断言所需方向数≥c，不支持未经定义的零空间方向选择。特征求解使用FP64，最后返回FP32权重；其余公式不变。较小矩阵直接求特征。c按输出维数截断。
5. 零增量不做0/0范数恢复，完全投影成零时保留零。所有模型检查有限值和原Model严格加载。
6. 每层所有投影/mask均由原增量计算，再按固定任务顺序作用于其他任务的增量；最终是base + alpha * sum(trimmed deltas)，五任务共用同一份权重。

## 评测和判断

原URS五任务各1000例，N100，seed3407，greedy，POMO（CVRPB80、其他100），8倍增强，batch125。原DataFinder/DataReader/Tester及参考值不修改。主指标为五任务aug_gap等权均值，越低越好；同时保存逐任务gap、相对旧base百分点、改善任务数、三次重复均值/std。各配置按三次重复平均比较，不能挑最佳seed冒充整体结果。

先核对算法与上游线性分支及论文矩阵公式，完成两个默认模型的小规模GPU前向，再启动41项完整评测。输出原始JSON、CSV、结果报告、模型和必要实现核查记录。没有额外训练或优化器更新。
