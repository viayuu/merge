# 四卡执行矩阵建议（只读梳理）

日期：2026-11-16。用户当前已授权完成正式实验并持续监控，使用2×RTX4090和2×RTX3090；覆盖先前“仅写代码、不部署”的范围。本笔记未启动GPU任务，未修改实现或数据。

## 最小完整矩阵：45个唯一模型-任务组合

任务集合：

- T2 = `cvrp,cvrptw`
- T4 = `cvrp,cvrpb,cvrptw,ocvrp`
- T11 = `tsp,atsp,cvrp,acvrp,cvrpb,cvrptw,ocvrp,ocvrptw,op,pctsp,pdtsp`

约定两专家插值 `theta(alpha)=(1-alpha)*theta_cvrp+alpha*theta_cvrptw`，所以CLI权重为 `--weights (1-alpha) alpha`。

| 系统 | checkpoint来源 | 测试集合 | 唯一组合数 |
|---|---|---|---:|
| 本地通用URS | `URS/pretrained/unified_checkpoint_500.pt` | T11 | 11 |
| 原单任务专家 | `{task}_checkpoint_300.pt` | 每个专家自己的任务 | 11 |
| 两专家交叉对照 | CVRP专家→CVRPTW；CVRPTW专家→CVRP | 另2个交叉组合 | 2 |
| alpha=.25 | CVRP/CVRPTW权重=.75/.25 | T2 | 2 |
| alpha=.5 | 现有`checkpoints/average_2.pt`，来源为上述两专家 | T2 | 2 |
| alpha=.75 | CVRP/CVRPTW权重=.25/.75 | T2 | 2 |
| 四专家均值 | T4各自专家等权 | T4 | 4 |
| 十一专家均值 | T11各自专家等权 | T11 | 11 |
| 总计 | 17个不同checkpoint，其中5个merge模型 | 45个唯一组合 | 45 |

alpha=0/1直接复用原两个专家，既不重复生成checkpoint，也不重复评测。本地U在T2/T4上的结果由其T11结果子集复用；原专家本任务成绩也跨阶段复用。论文11条URS/STL成绩直接读取 `merge_lab/paper_results.csv`，没有“论文baseline需要GPU重跑”的额外job。

若希望补完整T4专家4×4交叉能力表，在已有T4的4条对角+2条CVRP/CVRPTW交叉上补10条即可，总55个唯一组合。当前实施方案没有强制这张全交叉表，因此45是满足主线的最小配置。TSP/ATSP双专家融合为旧报告的额外诊断，当前精简实施方案未列必跑；不应为凑GPU任务暗中扩大。

## 统一原论文设置

固定 `--test_scale_list 100 --test_episodes 1000 --eval_type greedy --seed 3407 --aug_factor 8 128`，不传 `--disable_aug`；原URS/dataset、原顺序前1000例、原DataReader及gap。原POMO由环境决定，保留原逻辑。

原 `--test_batch_size_small` 是 `1000 125 50 8`（对称/对称多仓/非对称/非对称多仓）。T11没有多仓；实际用对称1000、非对称50。四卡均24GB，建议先用原batch验证第一批真实正式job，不预设3090必须减batch。若发生实际OOM，再为对应任务统一降低各模型batch；不要只给某个模型或某张卡更小batch。非对称随机node IDs受seed及batch切分影响，固定二者，论文数字与本地复跑分别报告。

每个模型-任务用独立子进程，可让该任务在所有模型下均从seed3407起跑；这与把11任务串在单进程时后续任务的RNG推进不同。它保持原数据/解码算法与预算，属于本次固定的任务独立调用协议，需在最终记录说明，不能假定逐位等同论文的原训练/评测进程。

## 四卡粒度与依赖

最简单的动态队列单位为一个 `(checkpoint, task, N=100, episodes=1000, seed=3407, aug)`，每卡同一时刻一个进程。对称任务和非对称任务均可任意空闲卡执行；ATSP/ACVRP的128倍增强可优先交给4090，但不要为等待4090让3090长期空闲。首批可为U/CVRP、CVRP专家/CVRP、CVRPTW专家/CVRPTW、average2/CVRPTW，让四卡立即有工作，同时尽快得到B1可解释结果。

CPU合并5个模型是GPUjob的唯一文件依赖；average2已存在，其余4个可并行准备，但不覆盖旧权重。B1的10个专家/插值组合完成后优先分析；本地U其余任务、专家自任务等固定对照可提前排队，不依赖B1胜负。B2/B3均值实验在参数文件生成后即可排队，以小模型阶段先返回为优先级即可，无须空等整个wave或新增复杂调度框架。

每个job输出唯一目录，例如 `results/<run>/jobs/<model>__<task>/`，沿用 `evaluate.py --checkpoints <one> --out-dir <jobdir> --problem_set <task>`。记录GPU型号/ID、起止时间、退出码、日志、JSON路径。只有退出0且JSON含该任务完整1000例/正确增强/有效参考时计完成；进程启动不计成功。使用tmux维持队列，已有其他GPU任务不能被杀掉。

45个组合对应45,000次实例评估；各有原POMO与8/128增强。当前无GPU实测，不能从CPU两例推测GPU小时。前4个正式job返回后按任务类别实际耗时更新剩余时间；异构GPU耗时分型号报告，不跨卡速度比较。

## 现有summary的最小适配

`summarize.py` 按每个输入JSON和规模独立汇总，这是上一轮为防止混入不同seed/实例数而刻意设计的。一次传入45个单任务JSON能得到完整逐行paper对照，但每行宏平均只含一个任务，不能自动得到各模型T2/T4/T11均值。

可不改通用summary：在本次结果分析中按明确的实验模型ID、固定配置及预定任务集合聚合其返回的rows；分别做T2/T4/T11表，分母精确为2/4/11，缺任一目标任务就显示未完成，不把部分任务平均标作完整集合。不要恢复“仅按checkpoint stem自动混合所有文件”的旧错误。另一选择是把同一checkpoint/同一配置的单任务JSON组装成一个标明来源的group JSON后再调用既有summary；需先确认任务无重复、配置一致。

OP使用原最大化gap，其他任务使用原最小化gap，不另写统一cost公式。所有paper差值为 `local_aug_gap - paper_gap` 的百分点；负数更好。原专家baseline与merge、论文URS和本地U都应同时保留；本地U自身与paper的偏差用来解释同环境复现差异，不用它覆盖paper列。

## 共同base核查结果

本次CPU只读抽查CVRP300和unified500：顶层仅 `epoch, model_state_dict, optimizer_state_dict, result_log`；optimizer含 `state,param_groups`，没有训练初始化路径、seed或配置。结果日志为tuple结构的指标日志。`URS/pretrained/`只有11个300epoch专家和1个500epoch通用模型；现有`checkpoints/`只有average2，没有init/base/epoch0文件。

`train.py`默认seed和`model_load.enable=False`是代码默认，不是发布专家的历史训练证据。optimizer末态不能唯一恢复初始权重。因此标准TA/TIES/DARE应标 `BASE_UNVERIFIED`，而不是把通用模型、平均权重或随机重建模型当作真实共同base。无需阻塞平均/插值的45个组合，也不在本轮默默新增共同起点重训。

## 结束与后续条件

先完成45个唯一组合，按T2/T4/T11逐任务分析“接近原URS”及“是否只是单专家已有跨任务能力”。平均/插值失败也要完整保留，不为追正结果改任务或增训练。B4的99个未见任务在当前方案是后续可选项：只有源任务保留有值得延伸的信号时再扩展，否则汇报其延后原因；TA/TIES/DARE缺base属于适用条件未满足，不伪造完成状态。
