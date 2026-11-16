# 从URS代码与论文判断Common Base是否成立

2026-11-16（Asia/Shanghai）。本次复核以原训练代码和论文Appendix H为主，重新读取11个发布专家的权重与optimizer，并核对已有初始化候选。没有启动新的性能实验。

**判断：可以继续做同组专家、重建随机初始化base的TA/TIES/DARE探索。不能把全部11个URS-STL当成已经验证来自同一个预训练模型，也不能将候选文件称为作者的历史epoch0。** 缺少作者发布的base文件不是必须全部停下的理由。

**1. 原代码实际如何初始化？**

| 证据 | 原位置 | 能说明什么 |
|---|---|---|
| seed默认3407 | `URS/args.py:23` | 默认随机种子固定；不直接证明每个发布checkpoint的实际命令 |
| 先调用seed_everything，再构造Trainer | `URS/train.py:125` | 默认模型构造从受控RNG状态开始 |
| model_load.enable=False | `URS/train.py:99` | 主路径不加载一个共同预训练checkpoint |
| 先设置默认CUDA张量类型，再Model(**model_params) | `URS/Trainer.py:74`、`:88` | 参数原本直接在GPU上初始化，不能用CPU初始化后搬到GPU替代 |
| Model参数不含训练任务名或列表 | `URS/train.py:58`、`URS/model/Model.py:29` | 同一环境、seed、构造顺序下，改变训练任务列表本身不改变初始化 |
| 数据载入和训练在模型构造之后 | `URS/Trainer.py:128`、`:233` | 任务数据生成、random.choice和rollout不会先消耗模型初始化的随机数 |

因此，代码支持“不同任务从同一个可重建的随机初始化开始”这一可能性，并不支持“URS-STL默认从已训练的通用URS微调”。前者也能定义delta，但与经典论文的预训练后微调设定不同。

我将最初完整发布版本aa179e8与当前版本比较：Model.py、Model_LIB.py、Trainer.py、args.py、train.py的AST除文件头外完全相同；seed_everything函数AST也完全相同。当前utils.py的差别来自已记录的源码备份修复。这些初始化结论不是近期重构造成的。

**2. 论文是否承诺了相同初始权重？**

[URS Appendix H.2](https://arxiv.org/html/2509.23413v3#A8.SS2)说明STL仅使用单任务数据，架构、优化器和训练设置与多任务模型相同；训练300epochs，在251epoch降学习率，多任务模型则为500/451。H.1说明多任务训练硬件为4090。

这些描述支持共用实现与训练配方，但没有提供逐专家seed、初始化checkpoint哈希、PyTorch/CUDA版本，或明确声明11个专家来自同一个保存的epoch0。论文Table14也没有列训练seed。不能从“相同架构和超参数”直接推出“每个初始化参数逐位相同”。

当前train.py仍默认训练11任务，单任务设置只留有注释；发布STL所用的完整单任务启动配置未包含在checkpoint里。因此默认3407是可验证的候选生成起点，而不是对所有历史命令的直接记录。

**3. 参数检查如何补上证据？**

本次重新读取了11个原专家和三个候选，并复算原权重的未更新随机张量以及候选差异坐标上的余弦，得到与已有审计一致的结果：

- 10个非PDTSP专家的未更新随机参数，分别有612864或614528个元素，与两个CUDA候选均精确相等；PDTSP的1664个未更新随机元素也精确相等。CPU候选不匹配。
- 两个CUDA候选总计4960487个参数，其中348160个坐标不同（约7.019%），全部位于4个decoder大矩阵；其余4612327个坐标相同。共同的未更新指纹无法区分这4个矩阵。
- 只看348160个有区分力的坐标，发布专家呈现以下两组；与对应候选cosine约0.85–0.91，与另一候选接近0。

| 推断初始化类型 | 专家 |
|---|---|
| A：当前4090上重建的类型 | TSP、ATSP、ACVRP、CVRP、CVRPB、OCVRP、OCVRPTW、OP、PDTSP |
| B：当前3090上重建的类型 | CVRPTW、PCTSP |

这里的4090/3090是本次候选参数类型的简称，不是对作者历史物理GPU的认定。组别是强统计推断；未获得作者完整初始权重，不能将局部精确指纹升级为完整谱系证明。末次Adam状态不能唯一反演60万步的全部优化历史。

因此，现有证据足以支持先做同组候选base的探索，不支持直接宣布11个专家共享一个已确认的完整base。

**4. 各方法现在能否继续？**

| 实验 | 判断 |
|---|---|
| 同A组CVRP＋OCVRP，以A候选为base的TA/TIES | 可以进行，标注“重建随机初始化base”，不声称经典预训练设定已满足 |
| 同B组CVRPTW＋PCTSP，以B候选为base | 可以作为另一组探索 |
| DARE＋TA/TIES | 可计算；先检验单专家小drop的能力保留，不能照搬LLM的90%–99%drop经验 |
| 11专家共用一个已验证历史base | 目前不成立 |
| 用unified_checkpoint_500作为所有专家base | 没有训练谱系支持，并引入已有多任务训练能力 |

[Task Arithmetic](https://arxiv.org/html/2212.04089v3#S5)明确其原实验只考虑同一预训练初始化后的微调模型；[TIES Algorithm1](https://arxiv.org/html/2306.01708v2#S3)以共同初始化计算delta；[DARE §4.6](https://arxiv.org/html/2311.03099v3#S4.SS6)给出使用不匹配backbone、大delta时失败的实证。URS属于从头训练后合并的移植场景，能运行这些公式不等于自动继承其原实验结论。

另外，TA的θ0＋λΣ(θi−θ0)在λ=1/K时严格等于已做的均值，base会抵消。这一点无需历史base，也不用新增一次相同配置评测。对其他λ、TIES和DARE，base会影响结果。

同组专家相对候选的delta/init范数比约0.727–0.955，不能假定属于LLM小幅SFT增量。DARE可先使用drop=0/.01/.05/.1做单专家检查；所有评测仍沿用原URS数据和原gap。

此前CVRP＋CVRPTW属于跨组组合。它提示第一轮均值实验混入了候选初始化类型的差异，值得改做同组CVRP＋OCVRP验证；它还不足以证明初始化差异就是全部退化的原因。

**5. 本地可用文件**

- A候选：`experiment_runs/20261116_common_base_audit/init_4090_seed3407.pt`。
- B候选：`experiment_runs/20261116_common_base_audit/init_3090_seed3407.pt`。
- 两候选各约20MB，均可strict-load。已有GPU重复构造核对382/382参数张量SHA256相同，证明候选可重建；metadata明确没有获得历史epoch0。这些模型文件仍由Git忽略。
- 本轮重新复算结果：[code_paper_crosscheck.json](code_paper_crosscheck.json)。已有完整指纹和组别证据：[RESULTS.md](RESULTS.md)、[expert_groups.csv](expert_groups.csv)、[evidence_verification.json](evidence_verification.json)。

本轮仅新增本文和code_paper_crosscheck.json；保留目录内已有用户编辑、候选模型及原45项性能结果。
