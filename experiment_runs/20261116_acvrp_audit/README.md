# ACVRP 论文与本地差异的针对性排查

日期：2026-11-16，Asia/Shanghai。用户要求解释 ACVRP 的论文 URS/STL 3.06%/2.01% 与上一批本地 3.041837%/1.999560% 的差异。

所有检查均使用原 ACVRP 1000例、N100、同一checkpoint、seed3407、greedy、POMO100、128倍增强、batch50和原gap。未更改URS源码、原数据、参考值或上一批45项结果。

最初控制：

| 控制 | 节点 | GPU | 模型 | 用途 |
|---|---|---|---|---|
| standalone | gpu03 | 两张3090各一模型 | Unified / ACVRP专家 | 原设置重复，检查可复现性 |
| standalone | gpu05 | 两张4090各一模型 | Unified / ACVRP专家 | 只改变GPU型号 |
| after_atsp | gpu05 | 两张4090各一模型 | Unified / ACVRP专家 | 同一进程ATSP→ACVRP，检查任务顺序 |
| rng_advanced | gpu03 / gpu05 | 各自allocation内GPU | Unified / ACVRP专家 | 仅消耗ATSP等量随机数，不运行ATSP，隔离随机状态影响 |

`trace_test.py` 调用原 test.py，只给原 `unified_node_position_construction` 加只读记录：每批的CUDA随机状态、节点标识摘要。记录不抽取额外随机数。`advance_test.py` 在第一次ACVRP节点标识生成前，额外调用20次 `torch.rand((50*128,100,1),device='cuda')`，模拟ATSP对CUDA RNG的消耗；属于因果诊断，不能冒充原设置。

原源码证据：

- `URS/utils/utils.py:35`：整个进程设seed一次。
- `URS/Tester.py:149`：逐任务执行；没有在任务边界重设seed。
- `URS/Tester.py:218`：1000例按batch50执行，共20批。
- `URS/Tester.py:232`、`URS/model/Model.py:77`：每批调用模型pre_forward，构造节点位置特征。
- `URS/model/Model_LIB.py:208`：非对称问题调用CUDA torch.rand生成节点标识，greedy选节点本身是argmax。
- `URS/env/UniVRPEnv.py:241`：128倍复制原距离和需求后，各复制件使用不同节点标识。
- `URS/data/DataReader.py:610`：ACVRP参考值来自原文件result向量均值，为2.1027939319610596，并非gap汇总时被替换。

原始结果在results/，trace在同目录 *.trace.json，完整原URS日志在logs/；launch.sh和launch_advance_3090.sh为复现入口。使用已有Slurm1475/1465、单allocation一个step、内部双worker，经tmux执行。

最终结论与所有数值在RESULTS.md（排查完成后生成）。
