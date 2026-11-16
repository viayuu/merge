# 实现正确性审查

**最终状态：APPROVE（修复后复核通过）。以下保留首轮 REQUEST CHANGES 的原始发现；本轮列出的 2 项 BLOCKING 和 1 项 NONBLOCKING 均已修复，复核证据见文末。**

- 日期：2026-11-16
- 审查 agent：`/root/implementation_review`
- 精确模型：unknown；精确 reasoning effort：unknown。当前上下文没有可靠暴露这两个运行属性，不据通用身份描述推定。
- 总体判断：**REQUEST CHANGES**。合并公式、原推理兼容性和论文数值没有发现阻断问题；汇总存在两个需要修正的结果可信性问题。下面只要求局部修改和少量针对性检查。
- 范围：先读取 `git -C URS diff -- test.py Tester.py`，再审查当前 `merge_lab/{merge,evaluate,summarize}.py`、`paper_results.csv/md`、`merge_lab/README.md`、实施方案和相关原始读取代码。本次未改实现，未运行 GPU 或完整实验。

## BLOCKING

### P2：同名模型的不同运行被混成“任务等权均值”

位置：`merge_lab/summarize.py:21`、`:77-81`。

运行身份被压缩成 checkpoint 的文件名 stem，汇总再只按这个 stem 分组。独立调用 `summarize.py` 合并不同批次的 JSON 时，同一个 checkpoint 的不同 seed、实例数或配置会进入同一组；不同目录下同名 checkpoint 也会混在一起。`len(group)` 随后被标成“覆盖任务/规模数”，即使多行实际上是同一个任务。

已用两个临时 JSON 复现：二者 checkpoint 都是 `average_2.pt`，任务均为 CVRP100；正式配置为 1000 例、gap=2，另一次检查为 2 例、gap=20。输出为：

```text
| average_2 | 2 | 11.000 | 1 | 0.190 |
```

这不是两个任务的等权均值，而是同一任务的两种评测配置混合。README 明确支持合并不同批次 JSON，因此不能只依赖 `evaluate.py` 单次调用的重名检查。

最小修复：按输入 JSON 区分运行，显示简洁运行标识，并分别计算每次运行的均值；不要自动合并同名 checkpoint。若希望跨 JSON 汇总同一轮的分片，只需在明确确认配置一致、任务不重复后再合并，当前没有必要新增这种功能。不同任务集合的均值应显示集合或直接不作并排均值比较。

最小检查：上面的两个 JSON 应得到两个独立运行汇总，各自覆盖任务数为 1，平均 gap 分别为 2 和 20；不能出现 11。

### P2：外部参考解缺失后的占位 gap 仍被标为可与论文比较

位置：`merge_lab/summarize.py:27-33`；相关导出为 `URS/test.py:166-177`。

`paper_comparable` 检查实例数、增强、greedy 和数据根目录，但不检查原 reader 是否实际拿到了所需参考解。原 `DataFinder` 允许找不到 solution，CVRP PKL 和 VRPMIX PKL 的 reader 随后使用 `oracle_score=1.0` 占位。导出仍会生成正常 row、`missing=[]`，新汇总会把这个占位基准下的 gap 与论文相减。

已验证两段具体行为：

- 原 CVRP 前 2 例读取：传原 solution 时参考值为 `16.526185989379883`；传 `solution_name=None` 时为 `1.0`。
- 临时 CVRP100 JSON 使用原 dataset 路径、1000 例、8 增强、greedy，但 `solution_path=None` 时，`summarize()` 返回 `paper_comparable=True`。

**当前发布数据没有触发这类缺失**：本次确认 CVRP、CVRPB、CVRPTW、OCVRP、OCVRPTW 的外部参考文件都在。问题是现有缺失处理分支会把不完整参考误报成完整比较，不是声称这些文件现在损坏。

最小修复：仅对这些依赖外部参考文件的原 PKL 任务，缺少 `solution_path` 时保留原输出，但将论文差值留空并明确报告参考解缺失。也可在 runner 开始前检查这 5 类任务的原参考文件。无需改原 gap、硬编码新参考值或添加校验框架。

不能笼统拒绝全部 `solution_path=None`：原 TSP TXT 内嵌路线，ATSP/ACVRP PT 内嵌 optimal，OP/PCTSP/PDTSP 的原 loader 使用原有内置参考常数；这些属于用户明确要求保留的原评测口径。

最小检查：一个无外部参考的 CVRP PKL row 不应计算论文差值；原 TSP TXT / OP PT row 仍按既有口径工作。

## NONBLOCKING

### P2：透传相对 `--data_dir` 会被改按 URS 子目录解析

位置：`merge_lab/evaluate.py:25-44`。

wrapper 解析 checkpoint 和输出目录时使用调用者 cwd，但把额外 `--data_dir` 原样交给工作目录为 `URS` 的子进程。README 让用户在 `/public/home/shiys/merge` 执行 wrapper；从该目录显式传 `--data_dir URS/dataset` 会解析到不存在的 `/public/home/shiys/merge/URS/URS/dataset`。默认绝对路径不受影响，所以不阻断默认原数据流程。

已用 mock subprocess 捕获命令并确认解析后路径不存在，没有启动模型。

最小修复：透传前仅将 `--data_dir` 的相对值按 wrapper 调用 cwd 转成绝对路径；覆盖 `--data_dir value` 和 `--data_dir=value` 两种标准写法即可。或者明确文档规定这一选项必须用绝对路径，保持入口简单。

## 已核对且无需扩展的部分

- Average 是归一化非负加权平均；TA 为 `base + scale * sum(expert - base)`；DARE 对 delta 随机丢弃后除以保留率；TIES 是完整 task vector 的 top-density、幅度选符号及同符号非零均值。精确平票为零已在实现和文档中明确。没有发现当前算法公式错误。
- Delta 方法要求显式 base；没有把通用 URS 自动充当共同底座。真实共同训练底座的来源仍应由调用者提供，当前不需要增加来源框架。
- `URS/Tester.py` 只给现有结果字典增加已计算值；`test.py` 在 `tester.run()` 后导出 JSON。diff 没有更改原解码、数据顺序、随机调用、参考解读取或 gap 计算；NaN/Inf 在 JSON 序列化处被拒绝。
- `paper_results.csv` 的 11 对 URS/URS-STL 数值逐项与本地原论文 `papers/2509.23413.txt` Table 2/3 核对一致，包括 PCTSP STL 的 `-0.09`。差值计算为本地百分数 gap 减论文百分数 gap，单位 pp、负值更好；小样本和关闭增强行不会自动计算论文差值。
- 模型参数、seed 和批大小沿用原 CLI，runner 对同轮各 checkpoint 传相同参数；无需引入新的随机输入缓存、路线检查器、数据划分或完整测试套件。
- 本次只运行临时 JSON 汇总、原数据前 2 例的读取以及 mock CLI 检查。数学端点/strict-load 检查由实施方已有记录提供，本审查没有冒称重新执行；未完成正式 1000 例实验，也未尝试 GPU 驱动修复。

## 修复后复核（同轮，2026-11-16）

最终判断：**APPROVE；当前审查范围内没有剩余阻断项或未处理建议。** 本次只复核首轮三项修复，没有扩大范围或新增测试套件。

1. **运行分组：已修复。** 当前 `summarize.py` 按 `(source_json, scale)` 分组，显示运行标识与任务集合。重新使用同名 checkpoint、同一 CVRP 任务的 1000 例 gap=2 和 2 例 gap=20 输入，得到两行独立均值 2.000 和 20.000；正式行论文差为 0.190 pp，小样本行不计算论文差值；不再出现 11.000。
2. **缺少外部参考解：已修复。** 当前 `test.py` 增加 `reference_available`，只对原来需要外部解的 5 类 PKL 任务检查 solution 路径；保留原内嵌/常量参考口径。`summarize.py` 将该布尔值导出到 CSV，并在缺失时显示未完成说明、留空论文差值。临时 JSON 检查确认：参考可用的正式 CVRP 行可比较，参考不可用的正式 CVRP 行不可比较，原口径参考可用的 OP 行仍可比较。源码复核确认该检查位于推理后的导出阶段，没有修改 DataReader 或原 gap。
3. **相对数据路径：已修复。** 当前 `evaluate.py` 直接用 argparse 接收 `--data_dir`，在切换子进程 cwd 前调用 `resolve()`。mock subprocess 分别检查 `--data_dir URS/dataset` 和 `--data_dir=URS/dataset`，两者均传出存在的绝对目录 `/public/home/shiys/merge/URS/dataset`。

此外，新增的批大小导出字段已改为 `configured_batch_size`，明确它是配置批大小而非每个尾批的实际大小。

验证边界保持不变：上述通过表示实现与这几项小型正确性检查通过，不表示已经完成 GPU 正式评测、11 任务 sweep 或达到研究性能目标。

## CPU smoke 首次失败后的局部补充审查

结论：**PASS；一行修复准确覆盖日志中的失败原因，总体 APPROVE 不变。**

本次只读取 `refine-logs/code_smoke_attempt1.log` 和 `URS/utils/utils.py` 的相关代码及 diff，没有新增测试或检查其他模块。日志显示原 checkpoint 已成功加载，CVRP/CVRPTW 数据与参考路径已找到；随后在 `test.py` 的 `copy_all_src(tester.result_folder)` 调用中抛出 `FileNotFoundError`，具体不存在的源路径为 `/public/home/shiys/merge/URS/_ops.py`。`tester.run()` 位于该调用之后，所以这次失败不是模型推理失败，也不能计作评测已运行。

`copy_all_src()` 由模块的 `__file__` 计算绝对路径，原条件仅检查路径前缀，不能证明该路径对应真实文件。新增的 `and os.path.isfile(src_abspath)` 使这类不存在的动态模块路径跳过源文件复制，同时仍复制满足原条件的真实文件。修改只影响日志目录的源码备份，不改变 checkpoint、数据、随机状态、解码或 gap，符合本次最小修复范围。

本补充结论只确认失败原因与补丁的对应关系；相同 smoke 的重跑结果由执行方另行记录，本审查没有宣称重跑已经成功。
