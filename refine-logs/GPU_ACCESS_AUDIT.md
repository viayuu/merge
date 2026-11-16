# GPU 访问入口审计

日期：2026-11-16（Asia/Shanghai）。本次仅做只读设备、Slurm allocation、SSH 和 Python 环境探测，没有更改驱动、访问控制或启动实验。

## 结论

用户的 **2 张 RTX 3090 与 2 张 RTX 4090 均存在、已分配，在首次探测时空闲**。之前失败发生在 `admin01` 管理/登录节点，该节点没有 `/dev/nvidia*` 或 `/proc/driver/nvidia`。不能把管理节点的 `nvidia-smi` 失败解释为计算节点无卡或驱动损坏。

当前 Codex shell 的主机为 `admin01`；系统提供 Slurm `/opt/gridview/slurm/bin/{srun,sinfo,squeue,scontrol}`。`nvidia-smi` 来自用户目录的包装脚本；在没有本机驱动的管理节点运行该工具，不会自动进入已有 GPU allocation。

## 当前用户已有资源

通过 `squeue -u shiys`、`scontrol show job -dd 1465`、`scontrol show job -dd 1475` 核实：

| Slurm job | 节点 | 类型 | 用户分配的宿主 GPU 索引 | allocation 内逻辑索引 | 现场状态 |
|---|---|---|---|---|---|
| 1465 | gpu03 | 2 × RTX 3090 24GB | 3、4 | 0、1 | 各约2 MiB，利用率0% |
| 1475 | gpu05 | 2 × RTX 4090 24GB | 0、1 | 0、1 | 各约2 MiB，利用率0% |

两项均为用户既有 RUNNING 的 SothisAI SSH 实例 allocation，CPU=10、host memory=20G，TimeLimit=UNLIMITED。无需创建新 allocation。

只读 SSH 到宿主检查发现，gpu03 的其他卡1/2、gpu05的其他卡3/4/5有其他用户进程；这些不属于本任务可用资源。实验应进入上述 allocation 并仅使用其逻辑0/1。

## 已验证访问命令

```bash
srun --jobid=1465 --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 --immediate=5 \
  bash -c 'hostname; printf "CUDA_VISIBLE_DEVICES=%s\n" "$CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader'

srun --jobid=1475 --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 --immediate=5 \
  bash -c 'hostname; printf "CUDA_VISIBLE_DEVICES=%s\n" "$CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader'
```

两命令均退出0，各只看到用户分配的两张卡；进入后 `CUDA_VISIBLE_DEVICES=0,1`。3090 显存24576MiB，4090显存24564MiB。

同一入口内执行 `source /public/home/shiys/miniconda3/etc/profile.d/conda.sh`、`conda activate easynco`、`cd /public/home/shiys/merge`，Python 探测返回：

```text
gpu03: torch=2.5.1+cu124, cuda_available=True, device_count=2,
       devices=[NVIDIA GeForce RTX 3090, NVIDIA GeForce RTX 3090]
gpu05: torch=2.5.1+cu124, cuda_available=True, device_count=2,
       devices=[NVIDIA GeForce RTX 4090, NVIDIA GeForce RTX 4090]
```

两边 `/public/home/shiys/merge`、`URS/pretrained/unified_checkpoint_500.pt`、`URS/dataset` 均可见。共享文件系统和 easynco 可以直接复用，不需要代码、数据或权重同步。

## 调度建议

复用两个现有 allocation，**每个 allocation 只建立一个 `srun` step，在该 step 内启动两个独立子 worker**。两个子 worker 分别设置 `CUDA_VISIBLE_DEVICES=0` 和 `CUDA_VISIBLE_DEVICES=1`，再调用原 URS `--cuda 0`。这使四个独立任务分别绑定四张用户卡。每个子 worker 的 `OMP_NUM_THREADS`、`MKL_NUM_THREADS` 可设为2；每个 allocation 总计10CPU。

本审计已验证设备发现与环境导入，没有启动 CUDA 数值实验。主实验进程随后已完成最小推理，结果见下文；不需再在 admin01 排查或安装 NVIDIA 驱动。

## 01:20 后续：同时建立两个 step 报 busy 的原因

主实验进程并行运行四项 sanity，其中每个 allocation 一项成功、一项在 step 创建时被拒绝。只读检查发现：

- `sanity_3090_1.log`、`sanity_4090_1.log` 已结束并写出结果 JSON；`1465.536`、`1475.1126` 已完成，分别运行1分16秒、2分11秒。初期空日志不代表卡死。
- `sanity_3090_0.log`、`sanity_4090_0.log` 分别报 `Unable to create step ... Requested nodes are busy`，尚未进入 Python/CUDA。
- 即使命令传 `-c2`，`sacct` 记录两个成功 step 均为 `AllocCPUS=10`、`AllocTRES=cpu=10,gres/gpu=2,mem=0,node=1`。

实际安装的是 `Slurm 20.11.8-1.5.1-20221206`。该版本本地说明 `/opt/gridview/slurm/share/man/man1/srun.1` 第1981–1983行明确写 `--overlap` 只共享 CPUs；第1077–1078行说明未给 step 显式 `--gres` 时，默认分配 job 的全部 GRES。因此每个独立 `srun` 都申请两张 GPU，第二个 step 无剩余 GPU可分配。`--overlap` 在本版本不能覆盖这一 GRES 冲突。

最小解决办法是上述“每 allocation 一个 step，step 内双 worker”。父进程入口可用：

```bash
srun --jobid=1465 --overlap -N1 -n1 -c4 --immediate=10 bash /absolute/path/to/two_worker_launcher.sh
srun --jobid=1475 --overlap -N1 -n1 -c4 --immediate=10 bash /absolute/path/to/two_worker_launcher.sh
```

这两行只描述入口，没有在审计中启动。另一种方法是每 step 显式申请 `--gres=gpu:1`，但须保留 Slurm 自动选择的 `CUDA_VISIBLE_DEVICES`，不能沿用硬编码的 allocation 内GPU索引；目前没有必要引入这种索引处理。
