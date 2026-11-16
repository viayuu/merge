# URS 模型合并实验

使用 `URS/dataset` 发布数据、原 `URS/test.py` / `Tester.py` 的解码和 gap。没有新增或拆分数据集，也没有重新实现模型、环境或评测循环。

三个脚本：`merge.py` 生成权重，`evaluate.py` 依次调用原测试入口，`summarize.py` 对照论文表格。URS增加结果字段和可选 `--output_json`，并修复源码备份误复制PyTorch虚拟文件的启动错误；原公式、数据和随机数调用顺序不变。

## 1. 激活环境并合并两个专家

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/merge

python merge_lab/merge.py \
  --checkpoints URS/pretrained/cvrp_checkpoint_300.pt URS/pretrained/cvrptw_checkpoint_300.pt \
  --output checkpoints/average_2.pt
```

默认等权平均。插值可加 `--weights 0.25 0.75`，权重对应 checkpoint 顺序，统一归一化。同一个合并模型用于所有测试任务。

四专家：CVRP、CVRPB、CVRPTW、OCVRP。十一专家可用原发布文件通配符（通用模型为500 epoch，不在这个模式内）：

```bash
python merge_lab/merge.py \
  --checkpoints URS/pretrained/*_checkpoint_300.pt \
  --output checkpoints/average_11.pt
```

## 2. 按论文设置评测

```bash
python merge_lab/evaluate.py \
  --checkpoints URS/pretrained/unified_checkpoint_500.pt checkpoints/average_2.pt \
  --out-dir results/two_tasks \
  --problem_set cvrp,cvrptw
```

默认每任务取原数据前1000例、N=100、greedy+POMO、对称8倍/非对称128倍增强。其余选项原样传给 `URS/test.py`，例如 `--cuda 0`、`--seed 3407`、`--test_batch_size_small 128 32 8 2`。保持参与对比的模型使用相同参数、任务顺序和batch设置；原非对称模型的随机node ID也沿用原入口的种子行为。

全11任务是runner的默认集合：

```bash
python merge_lab/evaluate.py \
  --checkpoints URS/pretrained/unified_checkpoint_500.pt checkpoints/average_11.pt \
  --out-dir results/eleven_tasks
```

原入口也能直接使用合并权重并导出JSON：

```bash
cd /public/home/shiys/merge/URS
python test.py --problem_set cvrp,cvrptw --test_scale_list 100 --test_episodes 1000 \
  --model_load ../checkpoints/average_2.pt --output_json ../results/direct.json
```

长任务使用 tmux。每次实验用不同的 `--out-dir`，避免覆盖自己的旧结果。当前交互shell位于admin01登录节点；GPU需通过已有Slurm allocation 1475（gpu05双4090）和1465（gpu03双3090）访问。已实现本次四卡队列，正式入口禁止CPU回退，见[运行说明](../experiment_runs/20261116_four_gpu/README.md)。

## 3. 输出与论文对照

- 每个checkpoint一个JSON，含原始未舍入objective、参考值、no_aug/aug gap、耗时、实际POMO/增强、数据路径和参数。
- `comparison.csv` / `comparison.md` 展示本次gap、论文URS/URS-STL gap、差值（百分点，负值更好），并给任务等权均值。
- [paper_results.csv](paper_results.csv) 的11行来自论文v3第6页Table 2/3；[来源与核对说明](paper_results.md)。URS-STL列对应每个任务自己的专家，是11模型参考系统。
- 论文成绩是发表值；本地重跑unified是另一个明确命名的结果行。不能把二者混写或用本地结果替换论文数值。
- 原数据、实例数、规模或增强不匹配时，保留原gap输出，论文差值留空。小样本的gap只用来检查流程；64例不是论文1000例成绩。
- 若需要外部参考解的原PKL任务缺少解文件，论文差值也留空并注明；原内嵌参考和内置常数仍按原口径使用。不同JSON运行、不同规模分别汇总，不把同名模型的重复运行混成任务均值。
- `no_aug_score` 在增强开启时是同一次推理第0个增强的结果；耗时属于整个增强pass。CPU检查的耗时不与论文RTX4090时间比较。

合并不同批次的JSON结果，无须重新运行模型：

```bash
python merge_lab/summarize.py results/two_tasks/*.json --out-dir results/two_tasks
```

## 4. Task Arithmetic / TIES / DARE

实现已提供，只有确认真实共同训练底座后才使用 `--base`，没有自动用通用URS代替底座。

```bash
python merge_lab/merge.py --method ties \
  --checkpoints URS/pretrained/cvrp_checkpoint_300.pt URS/pretrained/cvrptw_checkpoint_300.pt \
  --base /absolute/path/to/verified_common_base.pt \
  --density 0.2 --scale 1 --output checkpoints/ties_2.pt
```

TA：`--method task-arithmetic --scale 0.5`；DARE+TA：`--method dare-ta --scale 0.5 --drop-rate 0.1 --seed 0`。这两个例子的0.5是两专家的1/K。TIES使用每个完整task vector的top-density裁剪、幅度选符号及同符号非零均值；精确平票取0。DARE丢弃delta并按保留率缩放。

所有方法只保存模型参数及简洁来源信息，不合并optimizer。合并计算使用CPU FP32，输出可由原URS严格加载。没有新训练、自动调参服务或实验队列框架。

## 最小通路检查

正式比较之前可只取原文件前两例：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python merge_lab/evaluate.py \
  --checkpoints checkpoints/average_2.pt --out-dir results/smoke \
  --problem_set cvrp,cvrptw --test_episodes 2 --disable_aug --cuda -1
```

该命令显式使用CPU，仅检查权重→原评测→JSON/CSV链路；不据此判断motivation成立。
