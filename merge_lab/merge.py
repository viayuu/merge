"""Merge URS checkpoints on CPU; the output loads directly in URS/test.py."""

# 本脚本在 CPU 上合并多个 URS 模型检查点，输出可以直接交给 URS/test.py 加载。
# 检查点使用字典格式，模型参数存放在 "model_state_dict" 字段中。
# 合并仅处理模型状态，不合并优化器、学习率调度器等训练状态。

import argparse
import math
from pathlib import Path

import torch


# 支持的四种合并方法：
# - average：直接对多个模型的参数做加权平均，默认各模型权重相同。
# - task-arithmetic：先计算各模型相对共同基座的参数增量，再将增量相加。
# - ties：先保留各任务增量中绝对值最大的部分，再处理符号冲突并取平均。
# - dare-ta：先随机丢弃部分任务增量并重缩放，再按任务向量相加的方式合并。
METHODS = ("average", "task-arithmetic", "ties", "dare-ta")


def load_state(path):
    # 从检查点中读取模型状态字典；不在这里实例化网络或恢复训练进度。
    # weights_only=True 使用受限反序列化方式；safe_globals 额外允许加载 set 类型。
    # map_location="cpu" 将张量映射到 CPU，避免占用 GPU 显存。
    with torch.serialization.safe_globals([set]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint["model_state_dict"]


def merge_states(states, method="average", *, base=None, weights=None,
                 scale=1.0, density=0.2, drop_rate=0.1, seed=0):
    """Merge matching state dicts in FP32 and return CPU tensors.

    `weights` are normalized averaging weights. Delta methods require the
    actual common base; this function cannot establish checkpoint lineage.
    TIES retains top-density coordinates across each whole task vector.
    """
    # 参数说明：
    # states：待合并的模型状态字典列表，各模型必须具有相同的键和对应张量形状。
    # method：合并算法，取值见 METHODS。
    # base：增量方法所需的真实共同基座，即这些模型微调前共享的模型状态。
    #       参数名称与形状一致只能说明结构兼容，无法证明模型确实来自同一基座。
    # weights：仅用于 average 的非负权重，函数内部会将其归一化到总和为 1。
    # scale：合并后任务增量的整体缩放系数，仅影响三种增量方法。
    # density：TIES 保留的增量坐标比例，针对每个模型的完整任务向量计算。
    # drop_rate：DARE 随机丢弃增量坐标的概率。
    # seed：DARE 在 CPU 上生成随机掩码时使用的随机种子。
    # 返回值：位于 CPU 的模型状态字典；浮点张量恢复为第一个模型对应张量的类型。

    # 先检查算法及参数组合，避免把缺少基座的参数差值误当成有效任务向量。
    if not states or method not in METHODS:
        raise ValueError("Provide at least one state dict and a supported method")
    if method != "average" and base is None:
        raise ValueError(f"{method} requires an explicit common base checkpoint")
    if method != "average" and weights is not None:
        raise ValueError("weights are only used by average")
    if method == "average" and base is not None:
        raise ValueError("average does not use a base checkpoint")
    # 缩放系数允许为零或负数，但不能是 NaN 或无穷大。
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    if method == "ties" and not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    if method == "dare-ta" and not 0 <= drop_rate < 1:
        raise ValueError("drop-rate must be in [0, 1)")

    # 使用第一个模型作为结构模板，确定参数顺序、形状及最终保存的数据类型。
    # 如果传入基座，也将它纳入结构一致性检查。
    template = states[0]
    inputs = states + ([base] if base is not None else [])
    for state in inputs[1:]:
        # 状态字典的键集合必须完全相同，不能有缺失或额外的参数/缓冲区。
        if state.keys() != template.keys():
            raise ValueError("Checkpoint parameter keys differ")
        for key in template:
            # 同名张量的形状也必须一致，才能按相同坐标执行后续运算。
            if state[key].shape != template[key].shape:
                raise ValueError(f"Checkpoint parameter shape differs: {key}")

    # 只对浮点张量做数值合并；state_dict 中也可能包含整数计数器等非浮点状态。
    float_keys = [key for key, value in template.items() if value.is_floating_point()]
    merged = {}
    for key in template.keys() - set(float_keys):
        # 非浮点状态要求所有输入（包含基座）完全相同，否则无法直接保留其中一个。
        if not all(torch.equal(template[key], state[key]) for state in inputs[1:]):
            raise ValueError(f"Non-floating values differ: {key}")
        # 断开计算图并复制到 CPU，避免输出与原始模型共享张量存储。
        merged[key] = template[key].detach().cpu().clone()

    def flatten(state):
        # 按模板的固定键顺序，将所有浮点张量展平并拼成一个完整参数向量。
        # 统一使用 CPU 上的 FP32 计算，输出时再恢复模板的数据类型与形状。
        return torch.cat([state[key].detach().cpu().float().reshape(-1)
                          for key in float_keys])

    # 设模型数为 N、浮点元素总数为 P，则 vectors 的形状为 [N, P]。
    # 各模型中相同参数的相同元素，在这些向量中具有相同的列索引。
    vectors = torch.stack([flatten(state) for state in states])
    if method == "average":
        # 未指定 weights 时先使用 N 个 1，归一化后即为等权平均。
        coefficients = torch.tensor(weights if weights is not None else [1.] * len(states),
                                    dtype=torch.float32, device="cpu")
        # 权重数量必须与模型数一致，所有权重必须有限、非负且总和大于零。
        if (coefficients.numel() != len(states) or not torch.isfinite(coefficients).all()
                or (coefficients < 0).any() or coefficients.sum() <= 0):
            raise ValueError("weights must match checkpoints, be nonnegative and have positive sum")
        coefficients = coefficients / coefficients.sum()
        # coefficients[:, None] 将权重变为 [N, 1]，广播到各模型的全部参数坐标。
        # 沿模型维度求和，得到 result = Σ_i 权重_i × 模型参数_i。
        result = (vectors * coefficients[:, None]).sum(dim=0)
    else:
        # 三种增量方法都先构造任务向量：delta_i = 模型参数_i - 基座参数。
        # origin 的形状为 [P]，广播后 deltas 的形状为 [N, P]。
        origin = flatten(base)
        deltas = vectors - origin
        if method == "ties":
            # TIES 第一步：裁剪每个任务向量，只保留绝对值最大的部分坐标。
            # 保留范围是整个模型的浮点参数向量，并非每一层单独保留相同比例。
            if density < 1:
                # 保留数向下取整且至少为 1；density=1 时跳过裁剪。
                count = max(1, int(deltas.shape[1] * density))
                indices = deltas.abs().topk(count, dim=1, sorted=False).indices
                trimmed = torch.zeros_like(deltas)
                # gather 取出被选中的原始增量，scatter_ 将它们写回全零向量的对应位置。
                # topk 使用绝对值排名，但写回时保留增量原本的正负号。
                trimmed.scatter_(1, indices, deltas.gather(1, indices))
                deltas = trimmed
            # TIES 第二步：逐坐标对裁剪后的增量求和，以总和的符号作为选定方向。
            # 这里由带幅值的总和决定方向，并非只比较正号和负号的数量。
            elected = deltas.sum(dim=0).sign()
            # TIES 第三步：仅保留与选定方向一致、且非零的任务增量。
            selected = (deltas.sign() == elected) & (deltas != 0)
            # 每个坐标按实际保留下来的任务数取平均，分母至少为 1 以避免除零。
            # 若增量总和恰好为零，选定符号也为零；该坐标没有符合条件的非零项，更新为零。
            update = (deltas * selected).sum(dim=0) / selected.sum(dim=0).clamp_min(1)
        else:
            if method == "dare-ta" and drop_rate > 0:
                # DARE：为每个任务的每个坐标采样保留掩码，保留概率为 1-drop_rate。
                # 使用独立的 CPU 随机数生成器，使固定种子下的掩码可复现，
                # 同时不修改 PyTorch 的全局随机数生成器状态。
                generator = torch.Generator(device="cpu").manual_seed(seed)
                keep = torch.rand(deltas.shape, generator=generator, device="cpu") >= drop_rate
                # 被丢弃的增量置零，保留的增量除以保留概率，使单个坐标的期望保持不变。
                deltas = deltas * keep / (1 - drop_rate)
            # task-arithmetic 直接累加任务向量；dare-ta 累加随机丢弃并重缩放后的向量。
            # 此处使用求和，不除以模型数；dare-ta 在 drop_rate=0 时也走直接求和。
            update = deltas.sum(dim=0)
        # 将缩放后的合并增量加回共同基座：最终参数 = 基座参数 + scale × 合并增量。
        result = origin + scale * update

    # 将长度为 P 的结果向量拆回各个参数张量，offset 记录当前已读取的元素数。
    offset = 0
    for key in float_keys:
        value = template[key]
        # 按模板张量的元素数切片，恢复形状和数据类型，并复制为独立存储。
        merged[key] = result[offset:offset + value.numel()].reshape(value.shape).to(value.dtype).clone()
        offset += value.numel()
    # 非浮点张量此前已复制完成；这里将完整状态字典按模板的原始键顺序返回。
    return {key: merged[key] for key in template}


def main():
    # 命令行入口：解析输入路径和合并配置，然后加载、合并并保存模型状态。
    parser = argparse.ArgumentParser(description=__doc__)
    # --checkpoints 接收一个或多个检查点路径；--output 指定输出检查点文件。
    parser.add_argument("--checkpoints", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", choices=METHODS, default="average")
    # --base 仅供增量方法使用，必须由调用者明确提供真实的共同基座。
    parser.add_argument("--base", type=Path, help="Actual common base for delta methods; no default")
    # --weights 按检查点顺序提供，只用于 average；无需预先归一化。
    parser.add_argument("--weights", nargs="+", type=float, help="Averaging weights, normalized to sum to 1")
    # --scale 控制增量强度；--density 控制 TIES 保留比例。
    parser.add_argument("--scale", type=float, default=1., help="Scale of the merged task delta")
    parser.add_argument("--density", type=float, default=.2, help="TIES retained fraction (default: 0.2)")
    # --drop-rate 和 --seed 分别控制 DARE 的丢弃概率与随机掩码种子。
    parser.add_argument("--drop-rate", type=float, default=.1, help="DARE delta drop probability")
    parser.add_argument("--seed", type=int, default=0, help="DARE mask seed")
    args = parser.parse_args()

    # 在读取模型前提示缺失的基座；未知真实基座时，不能用 URS 通用模型随意替代。
    if args.method != "average" and args.base is None:
        parser.error(f"--base is required for {args.method}; do not substitute URS generalist for an unknown base")
    # 按用户提供的顺序读取模型，该顺序也对应 average 的各项权重。
    states = [load_state(path) for path in args.checkpoints]
    base = load_state(args.base) if args.base is not None else None
    merged = merge_states(states, args.method, base=base, weights=args.weights,
                          scale=args.scale, density=args.density,
                          drop_rate=args.drop_rate, seed=args.seed)
    # 将来源路径和有效的算法配置与模型一起保存，便于后续追踪本次合并。
    # 路径记录为绝对路径；与当前方法无关的配置记为 None。
    info = {
        "method": args.method,
        "checkpoints": [str(path.resolve()) for path in args.checkpoints],
        "base": str(args.base.resolve()) if args.base is not None else None,
        # 平均方法记录归一化后的实际权重；未指定时记录等权值。
        "weights": ([value / sum(args.weights) for value in args.weights]
                    if args.weights is not None else [1 / len(states)] * len(states))
                   if args.method == "average" else None,
        "scale": args.scale if args.method != "average" else None,
        "density": args.density if args.method == "ties" else None,
        "drop_rate": args.drop_rate if args.method == "dare-ta" else None,
        "seed": args.seed if args.method == "dare-ta" else None,
        # 记录 TIES 的零符号处理约定与合并过程的计算精度。
        # merge_dtype 指中间运算精度，保存张量的类型由第一个模型的对应张量决定。
        "ties_zero_sign": "zero",
        "merge_dtype": "float32",
    }
    # 创建输出目录，保存可供 URS/test.py 读取的模型字段及额外的合并元信息。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": merged, "merge_info": info}, args.output)
    print(f"Saved {args.method}: {len(states)} checkpoints -> {args.output}")


if __name__ == "__main__":
    # 直接运行脚本时执行命令行流程；被其他模块导入时仅提供函数定义。
    main()
