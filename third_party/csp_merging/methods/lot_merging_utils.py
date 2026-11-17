import torch
import copy
import torch.nn.functional as F

from datasets.common import get_dataloader
from datasets.registry import get_dataset
from datasets.MyImageNet import SequentialImagenet
from functools import partial

from collections import defaultdict
import numpy as np


#### LoTMerging 原版：OPT Merging 非正式版本，实现有一点问题，ln层没考虑


def quantile_via_sorting(tensor, q):
    """
    使用排序计算分位数
    Args:
        tensor: 输入张量（多维或一维）
        q: 分位数，例如 0.9 表示 90% 分位数
    Returns:
        分位数值
    """
    # 将张量展平为一维
    flat_tensor = tensor.view(-1)

    # 排序张量
    sorted_tensor, _ = torch.sort(flat_tensor)

    # 获取对应分位数的位置索引
    index = int(q * (sorted_tensor.numel() - 1))

    # 返回分位数值
    return sorted_tensor[index]


# clip vector version
def lot_merging(args, task_vectors, pretrained_checkpoint, exam_datasets, cur_task_vectors=None):
    if cur_task_vectors is None:
        cur_task_vectors = copy.deepcopy(task_vectors)
    opt_tv = copy.deepcopy(sum(task_vectors))

    # tackle the bias
    model = cur_task_vectors[0].apply_to(pretrained_checkpoint, scaling_coef=1).to('cuda')
    for name, pp in list(model.named_parameters()):
        opt_tv.vector[name] = opt_tv.vector[name] / len(task_vectors)

    feat_arr = []

    for t_id, exam_dataset in enumerate(exam_datasets):
        tv = cur_task_vectors[t_id]
        model = tv.apply_to(pretrained_checkpoint, scaling_coef=1).to('cuda')

        dataset = get_dataset(
            exam_dataset,
            model.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size
        )
        dataloader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=None)

        # 用于保存参数输入特征的字典
        accumulated_features = defaultdict(list)
        accumulated_activations = defaultdict(list)

        def hook_fn(name, module, input, output):
            # 将输入特征移到 CPU 并累加
            accumulated_features[name].append(input[0].detach().cpu())  # input 是元组，取第一个元素
            accumulated_activations[name].append(output[0].detach().cpu())  # input 是元组，取第一个元素

        hooks = []
        for name, param in model.named_parameters():
            if 'ln' in name and 'weight' in name:
                if 'ln_post' in name:
                    continue
                # 获取参数所在的模块路径
                module_name = ".".join(name.split(".")[:-1])  # 排除参数本身名称
                if module_name:  # 确保有模块
                    # 获取模块对象
                    module = dict(model.named_modules())[module_name]
                    # 注册钩子
                    hook = module.register_forward_hook(partial(hook_fn, name))
                    hooks.append(hook)

            elif name == 'model.visual.class_embedding' \
                    or name == 'model.visual.conv1.weight' \
                    or name == 'model.visual.positional_embedding' \
                    or name == 'model.visual.proj' \
                    or len(tv.vector[name].shape) == 1 \
                    or 'bias' in name:
                continue
            else:
                # 获取参数所在的模块路径
                module_name = ".".join(name.split(".")[:-1])  # 排除参数本身名称
                if module_name:  # 确保有模块
                    # 获取模块对象
                    module = dict(model.named_modules())[module_name]
                    # 注册钩子
                    hook = module.register_forward_hook(partial(hook_fn, name))
                    hooks.append(hook)

        exp_size = 0
        for j, data in enumerate(dataloader):
            inputs = data[0].to('cuda')
            if inputs.shape[0] > args.exp_size:
                inputs = inputs[:args.exp_size, ...]

            with torch.no_grad():
                model(inputs)

            exp_size += inputs.shape[0]
            if exp_size >= args.exp_size:
                break

        for hook in hooks:
            hook.remove()


        feat = {}
        model.to('cpu')
        for name, param in model.named_parameters():
            input_features = accumulated_features.get(name)
            output_features = accumulated_activations.get(name)
            if input_features is None:
                feat[name] = None
            elif 'ln' in name and 'weight' in name:
                if 'ln_post' in name:
                    continue

                output_features_ = torch.cat(output_features, dim=0)
                output_features_ = output_features_.view(-1, output_features_.shape[-1])  # bs * f_out

                weight = param
                bias_name = name.rsplit('.', 1)[0] if name.endswith('.weight') else name
                bias_name += '.bias'
                bias = model.state_dict()[bias_name]

                feat[name] = (output_features_ - bias) / weight
            else:
                feats = []
                for feat_ in input_features:
                    feats.append(feat_.view(-1, feat_.shape[-1]))
                input_feat_flat = torch.cat(feats, dim=0)
                feat[name] = input_feat_flat
        feat_arr.append(feat)

    for name, pp in list(model.named_parameters()):
        if opt_tv.vector[name] is None:
            continue
        elif feat_arr[0][name] is None:
            continue
        elif 'ln' in name and 'weight' in name:
            numerator = 0.0  # 分子初始化
            denominator = 0.0  # 分母初始化
            for t_id, tv in enumerate(cur_task_vectors):
                X_k = feat_arr[t_id][name]
                T_k = tv.vector[name]
                X_k_squared_sum = torch.sum(X_k ** 2, dim=0)
                numerator += X_k_squared_sum * T_k
                denominator += X_k_squared_sum  # 累加分母部分
            denominator = torch.where(denominator == 0, torch.tensor(1e-10), denominator)
            opt_tv.vector[name] = numerator / denominator

        else:
            # print(name)
            sum_XTX = 0.0
            for t_id, _ in enumerate(cur_task_vectors):
                X_k = feat_arr[t_id][name].to('cuda')
                sum_XTX += X_k.T @ X_k

            sum_XTT = 0.0
            for t_id, tv in enumerate(cur_task_vectors):
                X_k = feat_arr[t_id][name].to('cuda')
                T_k = tv.vector[name].to('cuda')
                sum_XTT += X_k.T @ X_k @ T_k.T

            # T_optimal = torch.linalg.solve(sum_XTX, sum_XTT)
            T_optimal = torch.linalg.pinv(sum_XTX) @ sum_XTT
            opt_tv.vector[name] = T_optimal.T.to('cpu')

    return opt_tv



def extract_feats(args, model, exam_dataset):
    dataset = get_dataset(
        exam_dataset,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )
    dataloader = get_dataloader(
        dataset, is_train=True, args=args, image_encoder=None)

    # 用于保存参数输入特征的字典
    accumulated_features = defaultdict(list)
    accumulated_activations = defaultdict(list)

    def hook_fn(name, module, input, output):
        # 将输入特征移到 CPU 并累加
        accumulated_features[name].append(input[0].detach().cpu())  # input 是元组，取第一个元素
        accumulated_activations[name].append(output[0].detach().cpu())  # input 是元组，取第一个元素

    hooks = []
    for name, param in model.named_parameters():
        if 'ln' in name and 'weight' in name:
            if 'ln_post' in name:
                continue
            # 获取参数所在的模块路径
            module_name = ".".join(name.split(".")[:-1])  # 排除参数本身名称
            if module_name:  # 确保有模块
                # 获取模块对象
                module = dict(model.named_modules())[module_name]
                # 注册钩子
                hook = module.register_forward_hook(partial(hook_fn, name))
                hooks.append(hook)

        elif name == 'model.visual.class_embedding' \
                or name == 'model.visual.conv1.weight' \
                or name == 'model.visual.positional_embedding' \
                or name == 'model.visual.proj' \
                or len(param.shape) == 1 \
                or 'bias' in name:
            continue
        else:
            # 获取参数所在的模块路径
            module_name = ".".join(name.split(".")[:-1])  # 排除参数本身名称
            if module_name:  # 确保有模块
                # 获取模块对象
                module = dict(model.named_modules())[module_name]
                # 注册钩子
                hook = module.register_forward_hook(partial(hook_fn, name))
                hooks.append(hook)

    exp_size = 0
    for j, data in enumerate(dataloader):
        inputs = data[0].to('cuda')
        if inputs.shape[0] > args.exp_size:
            inputs = inputs[:args.exp_size, ...]

        with torch.no_grad():
            model(inputs)

        exp_size += inputs.shape[0]
        if exp_size >= args.exp_size:
            break

    for hook in hooks:
        hook.remove()

    feat = {}
    model.to('cpu')
    for name, param in model.named_parameters():
        input_features = accumulated_features.get(name)
        output_features = accumulated_activations.get(name)
        if input_features is None:
            feat[name] = None
        elif 'ln' in name and 'weight' in name:
            if 'ln_post' in name:
                continue

            output_features_ = torch.cat(output_features, dim=0)
            output_features_ = output_features_.view(-1, output_features_.shape[-1])  # bs * f_out

            weight = param
            bias_name = name.rsplit('.', 1)[0] if name.endswith('.weight') else name
            bias_name += '.bias'
            bias = model.state_dict()[bias_name]

            feat[name] = (output_features_ - bias) / weight
        else:
            feats = []
            for feat_ in input_features:
                feats.append(feat_.view(-1, feat_.shape[-1]))
            input_feat_flat = torch.cat(feats, dim=0)
            feat[name] = input_feat_flat
    return feat


# 内存优化版本
def lot_mergingv2(args, task_vectors, pretrained_checkpoint, exam_datasets, cur_task_vectors=None):
    if cur_task_vectors is None:
        cur_task_vectors = copy.deepcopy(task_vectors)
    opt_tv = copy.deepcopy(sum(task_vectors))

    # tackle the bias
    model = cur_task_vectors[0].apply_to(pretrained_checkpoint, scaling_coef=1).to('cuda')
    for name, pp in list(model.named_parameters()):
        opt_tv.vector[name] = opt_tv.vector[name] / len(task_vectors)

    feat_XTX, feat_XTT = {}, {}
    for t_id, exam_dataset in enumerate(exam_datasets):
        print('Extracting features for task :', t_id)
        tv = cur_task_vectors[t_id]
        model = tv.apply_to(pretrained_checkpoint, scaling_coef=1).to('cuda')
        feat = extract_feats(args, model, exam_dataset)

        for name, pp in list(model.named_parameters()):
            if opt_tv.vector[name] is None:
                continue
            elif feat[name] is None:
                continue
            elif 'ln' in name and 'weight' in name:
                if name not in feat_XTX:
                    feat_XTX[name] = 0
                    feat_XTT[name] = 0

                X_k = feat[name]
                tv = cur_task_vectors[t_id]

                T_k = tv.vector[name]
                X_k_squared_sum = torch.sum(X_k ** 2, dim=0)
                feat_XTT[name] += X_k_squared_sum * T_k
                feat_XTX[name] += X_k_squared_sum  # 累加分母部分
            else:
                if name not in feat_XTX:
                    feat_XTX[name] = 0
                    feat_XTT[name] = 0

                X_k = feat[name].to('cuda')
                tv = cur_task_vectors[t_id]

                feat_XTX[name] += X_k.T @ X_k

                T_k = tv.vector[name].to('cuda')
                feat_XTT[name] += X_k.T @ X_k @ T_k.T


    for name, pp in list(model.named_parameters()):
        if opt_tv.vector[name] is None:
            continue
        elif name not in feat_XTX:
            continue
        elif 'ln' in name and 'weight' in name:
            numerator = feat_XTT[name]  # 分子初始化
            denominator = feat_XTX[name]  # 分母初始化

            denominator = torch.where(denominator == 0, torch.tensor(1e-10), denominator)
            opt_tv.vector[name] = numerator / denominator

        else:
            print('Calculating optimal task vector:', name)
            sum_XTX = feat_XTX[name]
            sum_XTT = feat_XTT[name]

            # T_optimal = torch.linalg.solve(sum_XTX, sum_XTT)
            T_optimal = torch.linalg.pinv(sum_XTX) @ sum_XTT
            opt_tv.vector[name] = T_optimal.T.to('cpu')

    return opt_tv


