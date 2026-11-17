# URS 五专家 CAT Merging

输入为原始URS第500轮多任务模型与五个已完成第30轮微调专家。本轮只合并与评测，两个正在续训的多任务模型不参与比较。

- [方案与论文/公开代码差异](PLAN.md)
- [独立实现审查](CODE_REVIEW.md)和[数值核对](implementation_checks.json)
- [结果](RESULTS.md)、[配置汇总](summary.csv)、[逐次逐任务结果](all_results.csv)
- manifest.json记录输入及协议；configs.json预设13种CAT配置；evaluation_plan.json安排39个CAT模型与旧base/TA两个对照。

运行环境为conda easynco，计算在Slurm allocation1465 / gpu03的2×RTX3090。现有4090续训不受影响。首次运行先完成prepare.py与小规模验证，然后：

```bash
tmux new-session -d -s urs_cat 'bash /public/home/shiys/merge/experiment_runs/20261117_cat_seen5/launch_all.sh > /public/home/shiys/merge/experiment_runs/20261117_cat_seen5/queue.log 2>&1'
```

脚本先按seed构建模型，再两卡并行调用原URS测试入口，全部完成后自动汇总。allocation编号是本次资源绑定，后续复现需替换为当时的授权资源。

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
python experiment_runs/20261117_cat_seen5/analyze.py --partial
```

每个配置重复3次，只改变从原数据选取的两个无标签实例；不训练、不另建数据集。比较配置使用三次平均，不选最优seed。所有输出为一份固定权重的多任务模型。模型和特征缓存由现有Git规则忽略。
