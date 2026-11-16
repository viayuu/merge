# ACVRP 为什么与论文相差一点？

2026-11-16，Asia/Shanghai。针对性复跑已完成：10个ACVRP评测进程，其中2个含ATSP前置任务；全部成功，原数据、checkpoint及评测参数不变。

**原因已定位：把每个任务拆成独立进程，改变了ACVRP随机节点标识所处的随机序列；换卡还会带来较小的差异。4090上ATSP→ACVRP的结果恢复为论文的3.06%/2.01%。**

**原表正确吗？**

正确。除ACVRP之外，10个任务的本地URS与本地专家结果保留两位小数均与论文吻合。ACVRP本地分别为3.0418372071%和1.9995601920%，保留两位为3.04%/2.00%；与论文3.06%/2.01%的差为−0.0181627929/−0.0104398080个百分点。

**原数据复跑得到什么？**

| 设置 | GPU | URS gap % | ACVRP专家 gap % |
|---|---|---:|---:|
| 论文 | 4090 | 3.06 | 2.01 |
| 单独ACVRP；与原本地结果逐位相同 | 3090 | 3.041837207 | 1.999560192 |
| 单独ACVRP；只换GPU | 4090 | 3.038288923 | 2.002190650 |
| 同一进程ATSP→ACVRP | 4090 | 3.060408014 | 2.006598935 |
| 只推进ATSP等量随机数，再跑ACVRP | 4090 | 3.060408014 | 2.006598935 |
| 只推进ATSP等量随机数，再跑ACVRP | 3090 | 3.054894256 | 2.005339263 |

全部使用同一原ACVRP数据1000例、N100、seed3407、greedy、POMO100、增强128、batch50。每次的参考均值完全相同，为2.1027939319610596。没有更换参考解、样本或checkpoint。

**为什么greedy仍受随机数影响？**

原[Model_LIB.py](/public/home/shiys/merge/URS/model/Model_LIB.py:208)对非对称任务使用`torch.rand`生成节点标识；greedy只固定动作选择为argmax，并没有固定模型输入中的随机标识。128份增强副本共享原距离和需求，但有不同随机标识。

原[test.py](/public/home/shiys/merge/URS/test.py:135)只在进程开始设置一次seed，[Tester.py](/public/home/shiys/merge/URS/Tester.py:144)按任务顺序继续使用同一CUDA随机数生成器。此前45项队列则为每个模型任务启动独立进程，每次重新seed；因此ACVRP从初始随机状态开始。

原[train_problem_list](/public/home/shiys/merge/URS/problem/ProblemSet.py:140)按名字长度稳定排序后为OP、TSP、ATSP、CVRP、ACVRP、PCTSP、CVRPB、OCVRP、PDTSP、CVRPTW、OCVRPTW。ACVRP之前只有ATSP使用随机节点标识。ATSP的1000例按batch50执行20批，每次生成(50×128,100,1)的随机标识；随后ACVRP接着这个随机序列生成自己的标识。

实测4090上，单独ACVRP与ATSP后ACVRP的20批节点标识摘要全部不同。只在ACVRP开始前消耗20次同形状torch.rand，不执行任何ATSP推理，得到的20批随机状态与节点标识摘要逐批完全等于ATSP→ACVRP；两个模型的增强/非增强objective及gap也逐位一致。这将差异定位到RNG推进，排除前一任务环境或解码器状态残留的解释。

**为什么其余任务没出现这个两位小数差异？**

对称任务使用固定坐标，不调用上述随机节点标识；ATSP是这11个任务里最先使用随机标识的任务，在4090上本次前置ATSP的URS结果仍为2.2581379294%，与原45项中的ATSP逐位一致。ACVRP是后续非对称任务，因此更容易暴露按任务重设seed与连续评测的区别。

**GPU型号有什么影响？**

同seed的单独ACVRP从3090换到4090，URS改变−0.003548284个百分点，专家改变+0.002630458个百分点。追踪确认，两种卡在首次特征构造前的CUDA RNG状态摘要相同，但生成的随机节点标识不同；20批摘要均不同。不能仅将其称为浮点运算误差。

[PyTorch 2.5.1的CUDA随机数实现](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cuda/DistributionTemplates.h#L44)按GPU的SM数量及每SM线程数设置生成网格和Philox counter推进量，因此相同seed和形状并不保证3090/4090生成逐元素相同的随机标识。这与本次追踪一致。

**对原实验有什么影响？**

上一批本地ACVRP数值是真实、可逐位重复的结果。它与本次按4090、ATSP→ACVRP执行所得到的论文精度结果，使用了不同随机标识。此前“沿用seed和batch即能完全对齐论文”的隐含假设不够；复现时还需记录GPU型号和进入任务前的CUDA RNG状态/任务顺序。

此前45项中URS、ACVRP专家和11专家平均在ACVRP上均用3090、相同seed、batch50且单任务进程，保持了这三者的输入随机性设置一致；ATSP的对应三者均使用4090。当前诊断不改变已经保存的负面合并结果。原CSV和JSON保留，不覆盖为论文数字。

后续若要按论文数字复现，可在4090上保持ATSP→ACVRP的原连续执行；若为了并行拆分任务，应显式记录并恢复相应任务的随机状态，同时固定GPU型号与batch。没有必要修改原数据集或gap。

论文没有提供用于表格的完整命令及RNG状态日志，因此不能断言作者历史运行必然采用该命令；这里已经用原代码建立了可重复的差异机制，并在两个checkpoint上同时复现了论文发表精度。

完整证据：[控制实验CSV](/public/home/shiys/merge/experiment_runs/20261116_acvrp_audit/comparison.csv)、[机器可读复核](/public/home/shiys/merge/experiment_runs/20261116_acvrp_audit/verification.json)、[原始结果目录](/public/home/shiys/merge/experiment_runs/20261116_acvrp_audit/results)、[运行说明](/public/home/shiys/merge/experiment_runs/20261116_acvrp_audit/README.md)。所有诊断脚本仅位于本运行目录，URS源码未新增修改。
