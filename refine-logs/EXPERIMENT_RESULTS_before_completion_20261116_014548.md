# 实验代码与初始检查结果

日期：2026-11-16。实施依据：[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。本轮用户要求完成代码，按原数据/原论文口径，遵循KISS/YAGNI。

## 已实现

- `merge_lab/merge.py`：平均/加权插值、Task Arithmetic、TIES、DARE+TA，CPU合并；delta方法必须显式提供base。
- `merge_lab/evaluate.py`：子进程调用原URS/test.py，默认原数据N100前1000例、8/128增强，保留原解码及gap。
- `merge_lab/summarize.py`：JSON转换CSV/Markdown，对照论文URS/STL，差值单位为百分点；按运行和规模分别汇总。
- `paper_results.csv/md`：11任务的原论文数字及来源。3个核心脚本共300行，没有新增训练、数据生成、调度或大型测试框架。
- 原URS仅增加结果字段/可选JSON输出，并对实际触发的源码备份虚拟文件错误增加文件存在性检查；不改DataReader、环境、模型或gap公式。

## 必要检查

合并端点、TA在scale=1/K时等于平均、TIES符号/分母/平票、DARE drop=0及seed重现均通过；真实CVRP+CVRPTW均值可strict-load原模型（4,960,487参数）。生成产物为 `checkpoints/average_2.pt`。

代码审阅发现并修复：不同运行同名checkpoint被混为任务均值；缺外部参考时错误计算论文差值；相对data_dir路径按子进程cwd解释。复核后APPROVE，详情见 [EXPERIMENT_CODE_REVIEW.md](EXPERIMENT_CODE_REVIEW.md)。

## B0：CPU通路检查 PASS

| 项目 | 实际设置/结果 |
|---|---|
| checkpoint | 原unified_checkpoint_500、average_2 |
| 任务 | CVRP、CVRPTW |
| 数据 | URS/dataset原文件前2例；N100；未新建或拆分 |
| 解码 | 原greedy+POMO，无augmentation，seed3407 |
| 设备 | CPU，OMP/MKL线程数2 |
| 完成情况 | 4个模型-任务组合，每组合2例，exit=0 |
| 产物 | 两份JSON、comparison.csv、comparison.md、原URS日志和Excel |
| 指标检查 | 原公式与JSON数值一致；小样本paper delta正确留空 |

这只证明权重→原评测→结果导出通路可用，不是论文设置复现或motivation结果。原始数值留在 `results/code_smoke/`，不使用2例无增强成绩比较论文1000例增强成绩。

## 两次准备问题及修复

1. `copy_all_src` 尝试复制PyTorch虚拟 `_ops.py`，推理前FileNotFoundError。仅在原文件拷贝条件加 `os.path.isfile`，已审阅。日志 `code_smoke_attempt1.log`。
2. 原评测推理已完成，但Excel导出缺少原README声明的openpyxl。给easynco安装 `openpyxl 3.1.5` 和其依赖 `et-xmlfile 2.0.0` 后原流程成功；未重写导出。日志 `code_smoke_attempt2.log`。

最终完整日志：[code_smoke.log](code_smoke.log)。复现命令在 [run_code_smoke.sh](run_code_smoke.sh)，本次通过tmux运行。Python编译及保留仓库CRLF的git diff whitespace检查通过。

## 正式实验状态

B1/B2/B3正式1000例实验均NOT_RUN。本轮范围为代码实现及必要检查；当前shell还未识别GPU（torch.cuda.is_available=False、device_count=0，nvidia-smi无法连驱动）。没有改驱动或启动整批实验，没有main-result positive/negative判定，因此没有触发额外ablation。

完整运行说明见 [merge_lab/README.md](../merge_lab/README.md)。
