# URS 本地只读审计（2026-11-16）

本审计只运行 CPU 权重加载/静态代码检查，未启动训练或模型推理，未修改 URS 源码。环境为 `conda easynco`，PyTorch `2.5.1+cu124`。以下事实在当前目录实际核实；训练历史未记录的部分保持未知。

## 核心结论

- URS 是官方 `CIAM-Group/URS`，论文为 *URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization*，arXiv `2509.23413`。README 当前标为 ICML 2026（本地 README 1–14、173–178）；代码 commit `69c6766219e40d18fd8b371fedfc208241e5be9e`，`git status --short` 为空。
- 本地有 11 个 specialist 和 1 个 unified checkpoint，所有模型 **382 个参数张量、4,960,487 个参数、0 个 buffer**，keys/shapes 完全相同，均可严格加载到同一个当前 `Model`。它们具备直接参数合并的架构条件。
- 11 个 specialist 均记录 epoch=300；unified epoch=500。不能仅凭 epoch 比较训练算力公平性，也不能把平均后的部署成本等同于所有专家的历史总训练成本。
- checkpoint 仅含 `epoch, model_state_dict, optimizer_state_dict, result_log`；没有初始化 checkpoint、seed 或训练命令。**共同训练起点未证实**，不可把 unified 当作单任务模型的真实共同 base。
- 数据资源已就绪：manifest 的 589 个数据文件全部在本地，另有 `dataset/README.md`，数据目录共 590 个文件。`.urs-setup/manifest.json` 保存下载来源、checkpoint SHA256 和压缩包校验记录；本审计未重算全部 SHA256。

## 结构、任务与合并边界

`URS/args.py:84–90` 默认 embedding=128、encoder=12 层、FFN=512、logit clipping=50、greedy。`URS/model/Model.py:31–37` 统一 encoder/decoder + 3/6/5 维位置/属性/节点类型 embedding，无需为不同任务替换网络形状。

`URS/problem/ProblemSet.py:140–142` 的 11 个训练任务为：`tsp, atsp, cvrp, acvrp, cvrpb, cvrptw, ocvrp, ocvrptw, pdtsp, op, pctsp`。代码共定义 110 个任务，99 个不在训练任务集合内（同文件 103–105）。

`URS/problem/ProblemRep.py:24–41` 使用 13 位约束/属性表示；`URS/model/Model.py:279–327` 的 decoder hypernetwork 由该表示生成 Wq/Wk/Wv/Wc；encoder 的 mixed bias 也受问题表示控制。可以接受已知任务属性，不能据此宣称模型无需任何任务条件。

`URS/model/Model.py:247–256`：relation 分支只在 PD 任务有 relation 矩阵时启用。将 11 个专家全参数均值时，需警惕特定任务更新被其他专家未更新参数稀释。对 CVRP/OCVRP 当前权重，124/382 张量逐元素完全相同；CVRP/TSP 为 112/382；例如非 PD 专家的 relation Wq/Wv 完全相同，而 CVRP/OCVRP 第 0 层 Wq_row cosine≈0.419。这是部分共同初始化/保持未更新的线索，**不是完整共同 base 的证明**。

`URS/model/Model_LIB.py:170–185` 是 affine InstanceNorm，`track_running_stats=False`；不存在 BatchNorm running-stat 重估这一必要步骤。

建议 merge 文件仅保存合并后的 `model_state_dict` 和独立 provenance，不平均 optimizer state、epoch 或历史 result_log。权重本体约 19.84 MB（float32），当前约 55–60 MB 的 checkpoint 含优化器与日志。

## 评测必须统一的事项

| 项目 | 当前实现与影响 |
|---|---|
| 载入 | `Tester.py:72–77` 使用同一个 `Model`，`load_state_dict` 默认 strict；合并产物可沿用该接口。 |
| greedy | `Tester.py:195–196,220–248` 为每步 argmax + POMO 多起点后取最优，不是单条 rollout。 |
| augmentation | `args.py:48–50` 默认对称 8，非对称 128；`--disable_aug` 仅关闭 augmentation，仍保留 POMO。 |
| POMO | `Tester.py:223` 请求 N；`UniVRPEnv.py:201–205` PD 截至 N/2、多 depot 用 N×depot；同文件 289 行 backhaul 还会减少起点。记录实际 POMO 数。 |
| mask | `Tester.py:202,240` 按任务注册固定 mask；`mask_registry.py:35–55` 组合 visited/op/pc/pd/vrp/tw/l/bp。所有模型使用相同 mask。 |
| 目标方向 | `Tester.py:250–260` 对 OP/AOP 保留正 prize，其余转为正 cost；282–287 对最大化用 `(reference-score)/reference`，最小化用 `(score-reference)/reference`。符号正确，不能把所有 reward 一律取负。 |
| 参考 gap | 目前是均值之比，不是逐实例 gap 均值（`Tester.py:282–287`）。应分别报告，避免混用。 |
| 逐实例证据 | 原 tester 只累计均值并保存 gap/time（`Tester.py:266–305`），没有完整 per-instance objective/route/feasibility 表。需评测 wrapper。 |
| 非对称随机性 | `Model_LIB.py:208–215` 即使 greedy 也生成随机 node IDs；固定模型间种子、任务顺序、批大小和数据顺序，最好缓存每例 ID，避免改变 batch 带来不同随机输入。 |
| 可行性 | mask 不能替代独立完整路线验证；`UniVRPEnv.py:745–753` 仅在部分 TSP 类终止时检查访问完整性。保存路线并另查约束、NaN、未终止实例。 |

所有方法使用相同实例、任务顺序、POMO/augmentation、精度、随机输入、batch size。初筛 N=100、greedy+POMO、no augmentation；最终确认再用对称 8-fold，非对称另列固定预算。

## 已核实的数据/参考解陷阱

11 个训练任务 N=100 都有数据。`DataFinder` 找不到单独 solution 文件并不等于没有参考解：TSP 的 txt 内有路线，ACVRP 的 pt 内有每例 `result`，ATSP 的 pt 只有聚合 `optimal=1.5711`。

| 任务 | 实际数据 | 参考信息 |
|---|---|---|
| TSP | `test_TSP100_n10000.txt` | 文件内路线，可按子集算参考成本。 |
| ATSP | `atsp100_n1000_seed1234.pt` | `dist` 为 (1000,100,100)，只有聚合 `optimal=1.5711`。 |
| ACVRP | `acvrp100_n1000_seed1234.pt` | `result` 有 (1000,) 每例结果，loader 支持切片。 |
| CVRP | `cvrp100_uniform.pkl` | `hgs_cvrp100_uniform.pkl`。 |
| CVRPB | `cvrpb100_uniform.pkl` | `ortools_400s_cvrpb100_uniform.pkl`。 |
| OCVRP | `ocvrp100_uniform.pkl` | `lkh_ocvrp100_uniform.pkl`。 |
| CVRPTW | `cvrptw100_1000uni.pkl` | `hgs_pyvrp_cvrptw100_1000uni.pkl`。 |
| OCVRPTW | `ocvrptw100_uniform.pkl` | `ortools_400s_ocvrptw100_uniform.pkl`。 |
| OP | `op100_n10000_seed1234.pt` | loader 硬编码 33.19，`DataReader.py:298`。 |
| PCTSP | `pctsp100_n10000_seed1234.pt` | loader 硬编码 5.98，`DataReader.py:355`。 |
| PDTSP | `pdtsp100.pkl` | 无外部 solution 时硬编码 9.428，`DataReader.py:500`。 |

`DataReader.py:188,270,327,385,534` 部分 pkl 参考解切片写成 `[start:total_episodes]`，而数据是 `[start:start+total_episodes]`。非零 start 的新 split 会错配；`DataReader.py:440,595,649,680,709` 部分 pt 解直接全文件均值。不能拿原 tester 打印的 gap 给自建小样本/划分下结论。

主指标宜用 **同实例 merged 相对 unified 的 objective 变化**；此比较不依赖所谓 optimal。对 OP/AOP 用 `(prize_U-prize_M)/max(|prize_U|,epsilon)`，其余用 `(cost_M-cost_U)/max(|cost_U|,epsilon)`，负数为 merged 更好。明确 epsilon 与零基线处理。外部参考 gap 只对真实逐例参考解与当前样本完全匹配时报告，OR-Tools/LKH 等统称参考解而非保证最优解。

## 训练起点、数据泄漏和单任务 CLI

- `train.py:42` 固定取得 11-task `train_problem_list`；`--add_training_problems`（43–45）只追加。`--problem_set cvrp` 不会把训练改成 CVRP 单任务，它只在 `test.py:set_problem_list` 被使用。`train.py:50` 的单任务列表仅为注释。若重训需新 wrapper 或明确 `--train_problem_set` 参数。
- `train.py:99–103` 默认不恢复权重；`Trainer.py:88` 新建模型。`train.py:125` 设 seed，但 checkpoint 无实际 seed/history，默认 seed=3407 不证明发布权重用了该值。
- 当前默认 validation 与 test 都读取同一个 `data_dir`，N=100、前 1000 例。`Trainer.py:132–138` 和 `Tester.py:156–162` 均不传 start；`DataReader.py:80` 默认 start=0。**当前代码默认重叠是已核实事实；发布权重历史是否实际用过这些样本仍未知。**
- 为 merge 校准单独生成固定 seed 数据；最终主评测用不同 seed 的新 held-out 数据；公开数据用于 secondary 的 paper-comparable 复现。校准与评测保留文件 hash/instance IDs。无需靠新实例参考最优解来算 merged vs unified 配对变化。
- 新数据生成有现成路径：`ProblemDef.py:25–115` 的 `get_random_problems` 包含上述 11 个源任务，CVRP 变种在 81–94 行共享生成器。该入口 41–42 行显式禁止 multi-depot，不能把“11-task 新测试集可生成”扩大为“全部 99 个 unseen 都可用此入口直接生成”。这里只静态核实路径，未实际生成或验证新分布与发布测试集一致性。
- 直接调用 ATSP 生成器时必须传 `problem_gen_params`；`UniVRPEnv.py:215–227` 原路径提供 `int_min=0,int_max=1000000,scaler=1000000`。新数据 wrapper 应复用此配置并记录容量与分布，避免新测试集意外改变问题分布。

## 建议的最小切片

1. CVRP 与 OCVRP 两任务：分别跑两 specialist、unified、0.5 权重平均；所有模型同一 N=100 held-out 样本。之后只在独立 calibration 扫 alpha={0,0.25,0.5,0.75,1}。两 endpoint 是 sanity/互相泛化 baseline。
2. 更异质的 TSP+CVRP 两任务重复；若第一对平均失败，不直接否定 merge，先查共同起点、尺度、对齐与模块冲突。
3. 扩展为 CVRP/OCVRP/CVRPTW/OCVRPTW 四专家；使用一组全任务共享的静态 merge 系数，禁止 test-task 专属系数，否则变为参数路由。
4. 11-task 合并与 99 unseen-task 外推分别报告。首轮 motivation 只要求合并保留源任务能力，不把 99-task zero-shot 设成第一道门槛。
5. 若真实共同 base 不能找回，Task Arithmetic/TIES/DARE 的标准 task-vector 设定不可声称已经满足。可以先做无需 base 的 averaging；共同初始化/共同 pretrained base 的对照属于后续受控重训分支，并单列其训练成本。

## CPU 核查复现要点

先 `source /public/home/shiys/miniconda3/etc/profile.d/conda.sh`，再 `conda activate easynco`。本地 checkpoint 的 `result_log` 包含 Python set，使用 `with torch.serialization.safe_globals([set]): torch.load(path, map_location='cpu', weights_only=True)`；不需要 unrestricted pickle 加载。

本审计对 12 个 checkpoint 逐个运行默认配置 Model 的 `load_state_dict(..., strict=True)`，均输出 `<All keys matched successfully>`。没有数值性能或 pilot 成功结论。
