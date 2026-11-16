# URS能否继续进行Task Arithmetic、TIES、DARE实验？

2026-11-16（Asia/Shanghai）。**可以有条件地继续：优先在同一候选初始化组内，用明确标注的“重建随机初始化base”做探索性实验；目前不宜把全部11个发布专家当作来自一个已验证的共同预训练模型。** 原“缺少作者base文件，所以所有delta方法都不能做”的结论过于笼统。

本次新增检查仅包括权重与优化器读取、CPU/两种GPU的初始化重建、精确指纹与参数统计，以及三种合并函数的加载检查。没有生成数据、训练模型或运行新的性能评测。

## 1. 已确认的源码与checkpoint事实

- 原[Trainer.py](/public/home/shiys/merge/URS/Trainer.py:88)直接构造Model；[train.py](/public/home/shiys/merge/URS/train.py:98)默认model_load.enable=False；[train.py](/public/home/shiys/merge/URS/train.py:125)在构造前seed_everything。默认seed=3407。这是从头训练的代码路径，不能据此认定发布模型全部实际使用默认命令。
- 11个专家都在epoch300、所有已建立optimizer state的参数step=600000；保存字段只有epoch/model_state_dict/optimizer_state_dict/result_log，没有初始完整权重、训练seed或物理GPU记录。通用URS是epoch500，其优化器各分支step不同，不能充当专家共同起点。
- 非PD专家的relation分支不参与前向，优化器也没有为这些参数建立state。10个非PDTSP专家分别留下72或73个非恒定随机张量（612864或614528个元素）；PDTSP留下未用的decoder.hyper_Wc.weight，共1664个随机元素。这里排除了全零偏置等平凡一致性。

## 2. 已经重建出有强指纹支持的候选base

使用原模型结构、构造顺序和seed3407，分别在CPU、4090和3090上创建未训练模型：

| 候选 | 10个非PDTSP专家的未更新随机张量 | PDTSP的未更新随机张量 |
|---|---|---|
| CPU初始化 | 全部不匹配 | 不匹配 |
| 4090上的CUDA初始化 | 每专家全部逐元素精确匹配 | 1664/1664元素精确匹配 |
| 3090上的CUDA初始化 | 每专家全部逐元素精确匹配 | 1664/1664元素精确匹配 |

这远强于仅依据“同架构、默认seed相同”的猜测：原权重留下的随机指纹被实际重现。但它不能单独证明完整初始模型唯一。

两种CUDA候选共有382个张量、4960487个参数；4个decoder hypernetwork大矩阵的348160个元素不同，占全模型约7.019%。每个矩阵不同87040个坐标：decoder.hyper_Wq_first.weight、hyper_Wq_last.weight、hyper_Wk.weight、hyper_Wv.weight。这些恰好不是上面的未更新指纹，因此两种完整候选均能通过局部指纹检查。

候选文件：[4090初始化](/public/home/shiys/merge/experiment_runs/20261116_common_base_audit/init_4090_seed3407.pt)、[3090初始化](/public/home/shiys/merge/experiment_runs/20261116_common_base_audit/init_3090_seed3407.pt)。文件各约20MB，由.gitignore排除。可用原Model strict-load，也可由当前merge.py的load_state直接读取。metadata明确historical_initial_checkpoint_obtained=False。它们是本次重建文件，不是作者给出的原始epoch0存档。

## 3. 发布专家强烈呈现两种候选初始化分组

只在两候选本来就不同的348160个坐标上，比较训练后的专家与初始候选的余弦相似度；没有用最终测试表现挑选base。

| 专家 | 与4090候选的cosine | 与3090候选的cosine | 推断候选组 | 未更新随机指纹精确匹配元素 |
|---|---:|---:|---|---:|
| ACVRP | 0.866671 | -0.001163 | 4090 | 612864 |
| ATSP | 0.851193 | -0.000058 | 4090 | 614528 |
| CVRP | 0.890520 | -0.000469 | 4090 | 612864 |
| CVRPB | 0.891602 | -0.000291 | 4090 | 612864 |
| CVRPTW | 0.000275 | 0.888406 | 3090 | 612864 |
| OCVRP | 0.896321 | 0.000629 | 4090 | 612864 |
| OCVRPTW | 0.893195 | -0.001561 | 4090 | 612864 |
| OP | 0.903004 | -0.000531 | 4090 | 612864 |
| PCTSP | -0.000097 | 0.903068 | 3090 | 612864 |
| PDTSP | 0.909065 | -0.000239 | 4090 | 1664 |
| TSP | 0.875074 | -0.000362 | 4090 | 614528 |

四个矩阵分别计算也支持相同分组，并非单个矩阵拉高均值。候选组为：

- A（4090候选）：ACVRP, ATSP, CVRP, CVRPB, OCVRP, OCVRPTW, OP, PDTSP, TSP。
- B（3090候选）：CVRPTW、PCTSP。

**证据层级：** 未更新随机张量相等是精确事实；“终点cosine约0.85–0.91 vs另一候选接近0”也是实测事实；据此推断各专家的完整初始权重更接近哪种候选，是强统计推断，不能代替作者训练日志或完整起点哈希。仍可能存在历史软件版本或其他训练前处理差异。4090/3090标签也不等于认定历史物理训练GPU，其他环境可能产生相同CUDA随机布局。

这些结果强烈反对不加验证地声称“全部11个专家共享完全相同初始权重”。默认seed相同不足以保证这一点。

## 4. 三类方法分别能否做？

| 方法 | 当前可进行的实验 | 应保留的限制 |
|---|---|---|
| Task Arithmetic | 同组专家、指定对应重建随机base，明确标注reconstructed-init adaptation；可改变整体delta缩放 | 原论文实证主要是同一预训练模型微调，不直接保证从头训练专家能叠加 |
| TIES | 同组、对应候选base，执行原trim/elect/disjoint-mean公式 | 裁剪的是相对候选base的delta；不能把任意全权重剪枝称作原设定 |
| DARE+TA/TIES | 同组、候选base，先检查单专家delta丢弃是否保持性能，再合并 | 当前从随机起点训练的delta很大，高丢弃率没有直接适用证据 |
| 全部11专家共用一个历史真实base的标准复现 | 暂不满足，尤其CVRPTW/PCTSP分组不同 | 应获得原始谱系证据或建立显式共享起点的受控训练支线 |

数学上，任何结构兼容的reference都能定义delta。缺少历史base并不让计算本身非法；它改变的是方法解释、对照设计及可支持的结论。若使用zero/平均专家/通用URS作为reference，应明确叫不同参考点的启发式实验。使用通用URS还会引入已经训练出的多任务能力，不适合作为“只合并单任务专家替代联合训练”的主证据。

### Task Arithmetic有一个无需base的特例

θmerge = θ0 + λΣᵢ(θᵢ−θ0) = λΣᵢθᵢ + (1−Kλ)θ0。

当λ=1/K时，θ0完全抵消，严格等于已做过的等权平均；该点不需要历史base，也不是一个新的独立baseline。对CVRP/OCVRP实测数值差异仅5.96e-8（FP32舍入）。其他λ或TIES的非线性裁剪/选符号、DARE的随机扰动，均不能据此消除reference的影响。

### DARE应按URS实际delta判断

按推断组选择对应候选后，11个专家的||delta||/||init||约为0.727–0.955。相对4090候选，多数同组专家的|delta|第95百分位约0.12–0.15；这与DARE论文考察的小幅SFT delta设定差别明显。不同架构的参数绝对尺度不能机械套阈值，但当前没有理由预期90%–99%的丢弃仍能保留能力。

DARE的倒数保留率缩放只保持参数扰动的期望，不保证非线性路由rollout的目标值期望。建议从单专家drop=.01/.05/.1开始，同时保留drop=0；再决定是否值得多专家合并。

## 5. 建议先做的最小方案

1. 首轮改为A组的CVRP+OCVRP，使用4090候选base。B2改为CVRP/CVRPB/OCVRP/OCVRPTW，四者均属A组。另一组CVRPTW+PCTSP可作独立对照。此前CVRP+CVRPTW以及四/十一专家平均都混入了两种候选初始化类型；这是潜在混杂因素，需要同组实验检验，不能直接宣布它就是平均失败的原因。
2. 同组对比平均、TA与TIES。TA可以λ=c/K、c=.5/1/1.5/2做小范围探索；c=1复用平均。TIES先density=.2、scale=1，必要时再少量改变。每个配置始终是一个固定模型，跨任务不改参数。
3. DARE先做上述单专家保真检查，再加入多模型矩阵；不先套LLM高drop率。
4. 沿用用户确定的原URS数据与参考分数、N100、1000例、原增强、原gap，不另划分校准/测试集。若在同一数据上扫描配置，完整报告为探索性结果。非对称任务还需固定推理GPU、batch和任务进入前RNG状态。
5. 当前可称“基于重建候选初始化的URS合并探索”。若要声称严格恢复了历史共同base，仍需作者提供完整初始化文件/哈希/训练环境，或用明确保存的共享起点重新建立受控专家。

## 6. 已完成的实现检查

4090候选base可strict-load原Model；对真实CVRP/OCVRP专家，TA（λ=.4）、TIES（density=.2、scale=1）、DARE+TA（drop=.1、λ=.5、seed0）均输出有限权重并可strict-load。未做性能推理，因此不能从这些检查宣称效果可行或优于URS。

## 一手来源与本地证据

- [Task Arithmetic §2和limitations](https://arxiv.org/html/2212.04089v3)：任务向量定义与同一预训练起点的原实验范围。
- [TIES §4–5](https://arxiv.org/html/2306.01708v2)：delta裁剪、选符号、同符号平均及固定超参配方。
- [DARE §3与§4.6](https://arxiv.org/html/2311.03099v3)：同源SFT delta、rescale以及错误backbone/大delta的失败例。
- [URS Appendix H](https://arxiv.org/abs/2509.23413v3)：单/多任务架构、训练设置与4090硬件描述；没有提供逐专家起点哈希。
- 本地[checkpoint清单](/public/home/shiys/merge/experiment_runs/20261116_common_base_audit/checkpoint_inventory.json)、[候选比较](/public/home/shiys/merge/experiment_runs/20261116_common_base_audit/candidate_comparison.json)、[分组CSV](/public/home/shiys/merge/experiment_runs/20261116_common_base_audit/expert_groups.csv)、[实现检查](/public/home/shiys/merge/experiment_runs/20261116_common_base_audit/implementation_check.json)。
- [4090候选指纹](/public/home/shiys/merge/experiment_runs/20261116_common_base_audit/init_4090_seed3407.json)、[3090候选指纹](/public/home/shiys/merge/experiment_runs/20261116_common_base_audit/init_3090_seed3407.json)、[CPU反例](/public/home/shiys/merge/experiment_runs/20261116_common_base_audit/init_cpu_seed3407.json)、[复现说明](/public/home/shiys/merge/experiment_runs/20261116_common_base_audit/README.md)。

候选重建与核查脚本全部在本运行目录；未修改merge_lab/merge.py已有用户编辑，也未覆盖原45项实验结果。
