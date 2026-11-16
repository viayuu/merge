# URS 模型合并实验

使用已发布的 URS 单任务模型，通过离线参数合并形成固定参数的多任务模型，并与原论文 URS 和本地复跑结果比较。

- [实验方案](refine-logs/EXPERIMENT_PLAN.md)
- [正式实验结果与分析](refine-logs/EXPERIMENT_RESULTS.md)
- [45项四卡实验结果](experiment_runs/20261116_four_gpu/summary.md)
- [ACVRP差异原因排查](experiment_runs/20261116_acvrp_audit/RESULTS.md)
- [合并与评测命令](merge_lab/README.md)

## 目录

| 路径 | 内容 |
|---|---|
| `URS/` | 上游 URS 源码及本项目的结果导出修改 |
| `merge_lab/` | 平均、插值、Task Arithmetic、TIES、DARE+TA和论文对照汇总 |
| `experiment_runs/` | 本次运行脚本、参数、日志、JSON/CSV、图表与核查记录 |
| `refine-logs/` | 实验方案、结果、审查与版本快照 |
| `papers/`、`idea-stage/` | 文献和前期调研材料 |

## 环境与数据

本机使用 `conda activate easynco`、PyTorch 2.5.1+cu124。正式评测使用原 URS 数据、参考值和 gap，N=100、每任务1000例，greedy，原8/128倍增强；运行细节见各实验目录。

模型和数据集不纳入 Git。复现前按 [URS说明](URS/README.md) 下载原文件，分别放入 `URS/pretrained/` 与 `URS/dataset/`。合并权重输出到 `checkpoints/`，同样由 `.gitignore` 排除。基于task delta的方法需要真实共同base，本次尚未运行这些方法。

仓库保留小型结果、日志、图表和文献PDF；排除Python缓存、原URS重复运行备份及临时锁文件。部分历史实验记录使用本机绝对路径，下载后可从以上相对链接查看报告，并按新环境调整运行路径。

## URS来源

上游项目：[CIAM-Group/URS](https://github.com/CIAM-Group/URS)，本地基于提交 `69c6766219e40d18fd8b371fedfc208241e5be9e`。本仓库将当前源码作为普通文件纳入版本管理，包含 `Tester.py`、`test.py` 与 `utils/utils.py` 的本地修改。保留上游 [AGPL-3.0许可证](URS/LICENSE)及版权声明。
