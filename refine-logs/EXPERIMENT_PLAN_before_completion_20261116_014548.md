# URS 单任务模型合并：实施方案

日期：2026-11-16。已按用户最新指令修改：**原数据、原评测、与原论文结果直接对照；不另建或拆分数据集。** 代码和必要检查已完成；本轮按用户授权执行2→4→11任务正式实验并持续监控、分析结果。

## 1. 固定研究问题

将发布的URS单任务checkpoint合并成一个参数量不增加的固定模型，在同一任务数据集与同一解码设置下接近或超过通用URS。先2任务、再4任务、最后11任务；99个未见任务后置。

主轨只使用 `URS/pretrained/*_checkpoint_300.pt` 的单任务专家，通用 `unified_checkpoint_500.pt` 作为论文与本地复跑对照，不混入merge或冒充共同base。保留URS原有任务表示、条件参数生成、环境mask，不按测试任务切换checkpoint或合并系数。

## 2. 数据与原评测口径

直接调用原 `DataFinder`、`DataReader`、`Tester`，使用 `URS/dataset` 已发布文件。**不生成新实例、不拆分校准集和测试集、不修改原参考解或硬编码参考均值。** 原代码默认的实例顺序与起始位置保持不变。

| 用途 | N | 每任务实例数 | 增强 | 对照用途 |
|---|---:|---:|---|---|
| 代码自检 | 100 | 少量（例如64） | 可关闭 | 只验证调用链与输出，不据此对比论文 |
| 正式论文对照 | 100 | 前1,000 | 对称8、非对称128 | 与Table 2/3的URS、URS-STL并排对照 |

原 `greedy` 仍含POMO多起点，沿用原设置。不同checkpoint复用同一任务列表、seed、batch size、数据顺序及增强配置；非对称任务即使greedy仍有随机node IDs，因此记录seed与批大小。本批固定seed=3407，先沿用原默认 `test_batch_size_small=1000 125 50 8`；每个任务独立进程从相同seed启动，不同checkpoint在相同任务上采用相同配置。当前不增加独立随机输入缓存或新的路线验证器。

主要输出沿用原评测值：`no_aug_gap`、`aug_gap`、总时间和每例时间；平均目标值若导出，应取原Tester计算值。正式比较使用 `aug_gap`。OP为收益最大化，其gap方向由原Tester处理；其余任务保持原cost口径。错误、NaN或不完整任务不能冒充完成结果。

论文对照数据在 `merge_lab/paper_results.csv`，来源为arXiv v3 PDF第6页Table 2/3。计算 `gap_delta_pp = local_aug_gap - paper_urs_gap_pct`，单位为百分点，负数更好；同时列出论文STL gap及对应差值。每任务分别报告，可对相同任务集合的gap差等权取均值，不能混合不同任务集合或把百分点写成相对百分比。

原计划中的新生成held-out、paired bootstrap及逐实例相对URS阈值不再用于本轮实施。代码只给出原口径结果和差值，不硬编码自动成功结论。论文值、本地通用URS复跑值和merge值分别标明来源。

## 3. 合并算法

| 方法 | 定义与首轮用法 | 前提 |
|---|---|---|
| Average / interpolation | 均匀平均；两模型全局权重对应alpha=0、0.25、0.5、0.75、1 | 不需要base；端点为原专家基线 |
| Task Arithmetic | `theta0 + scale * sum(theta_i - theta0)`；scale建议包含1/K | 必须显式提供真实共同base |
| TIES | task delta全局top-density裁剪、幅度选符号、同符号非零均值，再加回base；初始density=.2、scale=1 | 必须显式提供真实共同base |
| DARE+TA | delta按drop_rate随机丢弃、除以保留率，再TA；初始drop_rate=.1，保留drop=0检查 | 必须显式提供真实共同base，并记录mask seed |

checkpoint输出仅包含 `model_state_dict` 与必要合并信息，不平均optimizer或历史日志。每个方法使用一套固定权重服务目标任务集合。共同base未知时平均/插值照常可做，TA/TIES/DARE留待真实base到位；不因为CLI要求base就任意选通用URS。

第一轮只需均值和少量插值。若在原评测数据上探索超参，完整保留各配置结果并标为同数据上的探索性比较；不称为独立held-out确认。DARE多个mask分别对应不同固定模型，不按实例挑最好mask。

## 4. 实验顺序

| 块 | 任务 | 方法/对照 | 本轮状态 |
|---|---|---|---|
| B0 | CVRP、CVRPTW少量原实例 | 原checkpoint加载、均值checkpoint加载、原评测及CSV/JSON | PASS，已有检查结果 |
| B1 | CVRP+CVRPTW | 本地URS×2任务、两专家各测两任务、alpha=.25/.5/.75各测两任务；并列论文对照 | RUNNING，共12项 |
| B2 | CVRP/CVRPB/CVRPTW/OCVRP | 四专家均值×4任务；新增CVRPB/OCVRP的本地URS与本任务专家对照，复用B1重叠结果 | QUEUED，新增8项 |
| B3 | TSP/ATSP/CVRP/ACVRP/CVRPB/CVRPTW/OCVRP/OCVRPTW/OP/PCTSP/PDTSP | 十一专家均值×11任务；补余下7任务的本地URS与本任务专家对照，复用已有结果 | QUEUED，新增25项 |
| B4 | 其余99任务 | 已选合并checkpoint直接评测原发布数据 | DEFERRED，本批不执行 |

已完成最小调用链检查，本批直接执行45项唯一checkpoint×任务组合，均为原数据前1,000例和原增强配置。两专家插值定义为 `theta(alpha)=(1-alpha)*theta_CVRP+alpha*theta_CVRPTW`，alpha=0/1复用原专家结果。B1的跨任务专家对照用于区分单专家已有泛化与merge增益；B2/B3复用重叠对照，不重复评测同一checkpoint与任务。共同base仍未核实，TA/TIES/DARE不进入本批；不增加复杂机制或重训。

## 5. 最小代码与检查

复用原URS模型、环境、数据读取、参考分数、gap和推理循环，只增加合并CLI、评测结果导出和论文结果汇总。不要新建数据生成模块、任务调度框架、复杂配置层或大型测试套件。

必要检查限定为：合并输出能strict-load到原Model；单模型平均恒等/插值端点；TA在scale=1/K时与均值一致；DARE drop=0一致；TIES简单符号样例；原数据小样本能跑通且结果保存为JSON/CSV。保留错误信息，不针对假设中的异常写大量防御逻辑。

正式执行已获用户授权，45项队列已在 `tmux urs_merge_20261116` 启动，启动时4个worker处于RUNNING。硬件访问见 `GPU_ACCESS_AUDIT.md`：admin01为登录节点；Slurm allocation 1475在gpu05提供2×RTX4090 24GB，1465在gpu03提供2×RTX3090 24GB。每个allocation使用一个srun step，内部两个worker各占一张卡。默认环境为easynco；先按原默认batch执行，持续检查日志、退出状态与结果完整性。当前记录的是执行状态，尚无正式性能结论。
