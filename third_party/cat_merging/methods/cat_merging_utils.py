import torch
import copy
import torch.nn.functional as F

from datasets.common import get_dataloader
from datasets.registry import get_dataset

from collections import defaultdict
import numpy as np


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




# clip vector j version
def bal_mergingj(args, task_vectors, pretrained_checkpoint, exam_datasets, cur_task_vectors=None):
    if cur_task_vectors is None:
        cur_task_vectors = copy.deepcopy(task_vectors)
    result_task_vectors = copy.deepcopy(task_vectors)

    feat_arr = []

    for t_id, exam_dataset in enumerate(exam_datasets):
        tv = cur_task_vectors[t_id]
        model = tv.apply_to(pretrained_checkpoint, scaling_coef=args.scaling_coef_).to('cuda')

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

        # 定义钩子函数
        def hook_fn(module, input, output):
            # 将输入特征移到 CPU 并累加
            accumulated_features[module].append(input[0].detach().cpu())  # input 是元组，取第一个元素
            accumulated_activations[module].append(output[0].detach().cpu())  # input 是元组，取第一个元素

        hooks = []
        for name, param in model.named_parameters():
            # 获取参数所在的模块路径
            module_name = ".".join(name.split(".")[:-1])  # 排除参数本身名称
            if module_name:  # 确保有模块
                # 获取模块对象
                module = dict(model.named_modules())[module_name]
                # 注册钩子
                hook = module.register_forward_hook(hook_fn)
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

        # remove hooks
        for hook in hooks:
            hook.remove()

        feat = {}
        model.to('cpu')
        for name, param in model.named_parameters():

            module_name = ".".join(name.split(".")[:-1])
            if module_name:
                module = dict(model.named_modules())[module_name]
                input_features = accumulated_features.get(module)
                output_features = accumulated_activations.get(module)
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


                elif name == 'model.visual.class_embedding' \
                        or name == 'model.visual.conv1.weight' \
                        or name == 'model.visual.positional_embedding'\
                        or name == 'model.visual.proj'\
                        or len(tv.vector[name].shape) == 1\
                        or 'bias' in name:
                    # tv.vector[name] *= 0.0
                    continue
                else:

                    input_feat_flat = torch.cat(input_features, dim=0)
                    input_feat_flat = input_feat_flat.view(-1, input_feat_flat.shape[-1])  # bs * f_in

                    feat[name] = input_feat_flat
        feat_arr.append(feat)

    # eigenvalues_arr = []
    if args.ratio != 0:
        # calculate the removal basis for each task
        basis_arr = []
        for tv_id, tv in enumerate(result_task_vectors):
            basis = {}
            for name, pp in list(model.named_parameters()):
                if tv.vector[name] is None:
                    continue
                elif name == 'model.visual.class_embedding' \
                        or name == 'model.visual.conv1.weight' \
                        or name == 'model.visual.positional_embedding' \
                        or name == 'model.visual.proj' \
                        or len(tv.vector[name].shape) == 1 \
                        or 'bias' in name:
                    continue
                else:
                    feat_i = feat_arr[tv_id][name]    # bs f_in
                    if feat_i is None:
                        basis[name] = None
                        continue

                    xtx_i = torch.t(feat_i).mm(feat_i).to('cuda')  # f_in f_in
                    G = None
                    for tvj_id, tv_j in enumerate(result_task_vectors):
                        if tvj_id != tv_id:
                            feat_j = feat_arr[tvj_id][name].to('cuda')
                            xtx = xtx_i - args.lambd * torch.t(feat_j).mm(feat_j)   # f_in f_in


                            w_j = tv_j.vector[name].to('cuda')     # f_out f_in
                            G_ = w_j.mm(xtx).mm(torch.t(w_j))   # f_out f_out

                            if G is None:
                                G = G_
                            else:
                                G += G_

                    # 强制对称化（防止数值误差）
                    G = (G + G.T) / 2
                    eigenvalues, eigenvectors = torch.linalg.eigh(G)

                    if (eigenvalues <= 0).all():
                        basis_ = None
                    else:
                        positive_indices = (eigenvalues > 0)
                        positive_eigenvalues = eigenvalues[positive_indices]
                        positive_eigenvectors = eigenvectors[:, positive_indices]

                        if args.ratio == -1.0:  # pick positive basis
                            basis_ = positive_eigenvectors
                        elif args.ratio == 1 or args.ratio > 1:
                            r = args.ratio
                            basis_ = eigenvectors[:, 0:r]
                        else:
                            val_total = (positive_eigenvalues ** 2).sum()
                            val_ratio = (positive_eigenvalues ** 2) / val_total
                            r = np.sum(np.cumsum(val_ratio) < args.ratio)
                            basis_ = eigenvectors[:, 0:r]


                        basis_ = basis_.to('cpu')
                    basis[name] = basis_        # f_out r

            basis_arr.append(basis)

        for tvj_id, tv in enumerate(result_task_vectors):
            for tv_id, basis in enumerate(basis_arr):
                if tv_id != tvj_id:
                    for name, pp in list(model.named_parameters()):
                        if tv.vector[name] is None:
                            continue
                        elif name == 'model.visual.class_embedding' \
                                or name == 'model.visual.conv1.weight' \
                                or name == 'model.visual.positional_embedding'\
                                or name == 'model.visual.proj'\
                                or len(tv.vector[name].shape) == 1\
                                or 'bias' in name:
                            continue
                        else:
                            b = basis[name]  # f_out r
                            if b is None:
                                continue
                            b = b.to('cuda')
                            bbt_ = b.mm(torch.t(b))  # f_out f_out

                            vector_j = tv.vector[name].to('cuda') # f_out f_in

                            vector_orthogonal = (vector_j - torch.mm(bbt_, vector_j))

                            norm_original = vector_j.norm()
                            norm_remaining = vector_orthogonal.norm()
                            tv.vector[name] = (vector_orthogonal * (norm_original / norm_remaining)).to('cpu')


    ###################################################################
    ###############  tackle the weight of ln layers  ##################
    ###################################################################
    if args.ratio_ln != 0:
        for name, pp in list(model.named_parameters()):
            if 'ln' in name and 'weight' in name:
                if 'ln_post' in name:
                    continue
                if result_task_vectors[0].vector[name] is None:
                    continue
                if feat_arr[0][name] is None:
                    continue

                masks = []
                for tv_id, tv in enumerate(result_task_vectors):
                    out = 0
                    for tvj_id, tv_j in enumerate(result_task_vectors):
                        if tv_id != tvj_id:
                            out += torch.mean((feat_arr[tv_id][name] * tv_j.vector[name])**2, dim=0) - args.lambd_ln * torch.mean((feat_arr[tvj_id][name] * tv_j.vector[name])**2, dim=0)
                    topk_indices = torch.topk(out, args.ratio_ln, largest=True).indices
                    mask = torch.ones_like(out, dtype=torch.int)
                    mask[topk_indices] = 0
                    masks.append(mask)
                    # print(f"Parameter: {name}, mask: {mask}")


                for tv_id, tv in enumerate(result_task_vectors):
                    for tvj_id, tv_j in enumerate(result_task_vectors):
                        if tv_id != tvj_id:

                            mask = masks[tv_id]

                            original_norm = torch.norm(tv_j.vector[name], p=2)
                            masked_vector = tv_j.vector[name] * mask
                            masked_norm = torch.norm(masked_vector, p=2)

                            if masked_norm > 0:
                                tv_j.vector[name] = masked_vector * (original_norm / masked_norm)
                            else:
                                tv_j.vector[name] = masked_vector


    ###################################################################
    ###############  tackle the bias parameters  ##################
    ###################################################################
    if args.ratio_bias != 0:
        # tackle the bias of other layers
        for name, pp in list(model.named_parameters()):
            if 'ln' in name and 'weight' in name:
                continue
            elif name == 'model.visual.class_embedding' \
                    or name == 'model.visual.conv1.weight' \
                    or name == 'model.visual.positional_embedding' \
                    or name == 'model.visual.proj' \
                    or len(result_task_vectors[0].vector[name].shape) == 1 \
                    or 'bias' in name:
                params = []
                for tv in result_task_vectors:
                    params.append(tv.vector[name])

                stacked_params = torch.stack(params)

                # 用于存储每个任务的掩码
                masks = torch.ones_like(stacked_params)
                # 对每个任务计算掩码
                K = len(result_task_vectors)
                c = args.ratio_bias
                for i in range(K):
                    # 计算其他任务平方和
                    other_tasks_squared_sum = 0
                    for j in range(K):
                        if i != j:
                            other_tasks_squared_sum += result_task_vectors[i].vector[name] ** 2

                    # 找到平方和最大的 c 个分量的索引

                    flattened_abs_params = other_tasks_squared_sum.view(-1)
                    num_elements = flattened_abs_params.numel()
                    if args.ratio_bias == 1 or args.ratio_bias > 1:
                        k = num_elements - args.ratio_bias
                    else:
                        k = int(num_elements * (1 - args.ratio_bias))
                    threshold = torch.kthvalue(flattened_abs_params, k).values

                    mask_k = other_tasks_squared_sum < threshold
                    # print(threshold)
                    # print(other_tasks_squared_sum)
                    # print(mask_k)

                    # 在掩码数组中标记这些位置
                    for j in range(K):
                        if i != j:
                            masks[i] *= mask_k

                for j, tv in enumerate(result_task_vectors):

                    mask = masks[j]

                    original_norm = torch.norm(tv.vector[name], p=2)
                    masked_vector = tv.vector[name] * mask
                    masked_norm = torch.norm(masked_vector, p=2)

                    # if masked_norm > 0:
                    #     tv.vector[name] = masked_vector * (original_norm / masked_norm)
                    # else:
                    tv.vector[name] = masked_vector


    return result_task_vectors


# clip vector i j pair version
def bal_mergingij(args, task_vectors, pretrained_checkpoint, exam_datasets, cur_task_vectors=None):
    if cur_task_vectors is None:
        cur_task_vectors = copy.deepcopy(task_vectors)
    result_task_vectors = copy.deepcopy(task_vectors)

    feat_arr = []

    for t_id, exam_dataset in enumerate(exam_datasets):
        tv = cur_task_vectors[t_id]
        model = tv.apply_to(pretrained_checkpoint, scaling_coef=args.scaling_coef_).to('cuda')

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

        # 定义钩子函数
        def hook_fn(module, input, output):
            # 将输入特征移到 CPU 并累加
            accumulated_features[module].append(input[0].detach().cpu())  # input 是元组，取第一个元素
            accumulated_activations[module].append(output[0].detach().cpu())  # input 是元组，取第一个元素

        hooks = []
        for name, param in model.named_parameters():
            # 获取参数所在的模块路径
            module_name = ".".join(name.split(".")[:-1])  # 排除参数本身名称
            if module_name:  # 确保有模块
                # 获取模块对象
                module = dict(model.named_modules())[module_name]
                # 注册钩子
                hook = module.register_forward_hook(hook_fn)
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

        # remove hooks
        for hook in hooks:
            hook.remove()

        feat = {}
        model.to('cpu')
        for name, param in model.named_parameters():

            module_name = ".".join(name.split(".")[:-1])
            if module_name:
                module = dict(model.named_modules())[module_name]
                input_features = accumulated_features.get(module)
                output_features = accumulated_activations.get(module)
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


                elif name == 'model.visual.class_embedding' \
                        or name == 'model.visual.conv1.weight' \
                        or name == 'model.visual.positional_embedding'\
                        or name == 'model.visual.proj'\
                        or len(tv.vector[name].shape) == 1\
                        or 'bias' in name:
                    # tv.vector[name] *= 0.0
                    continue
                else:

                    input_feat_flat = torch.cat(input_features, dim=0)
                    input_feat_flat = input_feat_flat.view(-1, input_feat_flat.shape[-1])  # bs * f_in

                    feat[name] = input_feat_flat
        feat_arr.append(feat)

    # calculate the removal basis for each task
    basis_arr = []
    for tv_id, tv in enumerate(result_task_vectors):

        basis_arr.append([])
        for tvj_id, tv_j in enumerate(result_task_vectors):

            if tvj_id == tv_id:
                basis_arr[tv_id].append(None)
                continue

            basis = {}
            for name, pp in list(model.named_parameters()):
                if tv.vector[name] is None:
                    continue
                elif name == 'model.visual.class_embedding' \
                        or name == 'model.visual.conv1.weight' \
                        or name == 'model.visual.positional_embedding' \
                        or name == 'model.visual.proj' \
                        or len(tv.vector[name].shape) == 1 \
                        or 'bias' in name:
                    continue
                else:
                    feat_i = feat_arr[tv_id][name]    # bs f_in
                    feat_j = feat_arr[tvj_id][name]
                    if feat_i is None:
                        basis[name] = None
                        continue

                    xtx = torch.t(feat_i).mm(feat_i)  - args.lambd * torch.t(feat_j).mm(feat_j)   # f_in f_in
                    w_j = tv_j.vector[name]     # f_out f_in
                    G = w_j.mm(xtx).mm(torch.t(w_j))   # f_out f_out

                    # 强制对称化（防止数值误差）
                    G = (G + G.T) / 2
                    eigenvalues, eigenvectors = torch.linalg.eigh(G)

                    if (eigenvalues <= 0).all():
                        basis_ = None
                    else:
                        positive_indices = (eigenvalues > 0)
                        positive_eigenvalues = eigenvalues[positive_indices]
                        positive_eigenvectors = eigenvectors[:, positive_indices]

                        if args.ratio == -1.0:  # pick positive basis
                            basis_ = positive_eigenvectors
                        elif args.ratio == 1 or args.ratio > 1:
                            r = args.ratio
                            basis_ = eigenvectors[:, 0:r]
                        else:
                            val_total = (positive_eigenvalues ** 2).sum()
                            val_ratio = (positive_eigenvalues ** 2) / val_total
                            r = np.sum(np.cumsum(val_ratio) < args.ratio)
                            basis_ = eigenvectors[:, 0:r]
                    basis[name] = basis_
            basis_arr[tv_id].append(basis)         # f_out r
            # if basis is None:
            #     print(f"Parameter: {name}, basis None")
            # else:
            #     print(f"Parameter: {name}, basis shape: {basis_.shape}")

    for tvj_id, tv in enumerate(result_task_vectors):
        for tv_id, _ in enumerate(result_task_vectors):
            if tv_id != tvj_id:
                for name, pp in list(model.named_parameters()):
                    if tv.vector[name] is None:
                        continue
                    elif name == 'model.visual.class_embedding' \
                            or name == 'model.visual.conv1.weight' \
                            or name == 'model.visual.positional_embedding'\
                            or name == 'model.visual.proj'\
                            or len(tv.vector[name].shape) == 1\
                            or 'bias' in name:
                        continue
                    else:
                        basis = basis_arr[tv_id][tvj_id]

                        b = basis[name]  # f_out r
                        if b is None:
                            continue
                        bbt_ = b.mm(torch.t(b))  # f_out f_out

                        vector_j = tv.vector[name] # f_out f_in

                        vector_orthogonal = (vector_j - torch.mm(bbt_, vector_j))

                        norm_original = vector_j.norm()
                        norm_remaining = vector_orthogonal.norm()
                        tv.vector[name] = vector_orthogonal * (norm_original / norm_remaining)

    ###################################################################
    ###############  tackle the weight of ln layers  ##################
    ###################################################################
    if args.ratio_ln != 0:
        for name, pp in list(model.named_parameters()):
            if 'ln' in name and 'weight' in name:
                if 'ln_post' in name:
                    continue
                if result_task_vectors[0].vector[name] is None:
                    continue
                if feat_arr[0][name] is None:
                    continue

                masks = []
                for tv_id, tv in enumerate(result_task_vectors):
                    masks.append([])
                    for tvj_id, tv_j in enumerate(result_task_vectors):
                        if tv_id == tvj_id:
                            masks[tv_id].append(None)
                        else:
                            out = torch.mean((feat_arr[tv_id][name] * tv_j.vector[name])**2, dim=0) - args.lambd_ln * torch.mean((feat_arr[tvj_id][name] * tv_j.vector[name])**2, dim=0)
                            topk_indices = torch.topk(out, args.ratio_ln, largest=True).indices
                            mask = torch.ones_like(out, dtype=torch.int)
                            mask[topk_indices] = 0
                            masks[tv_id].append(mask)
                    # print(f"Parameter: {name}, mask: {mask}")


                for tv_id, tv in enumerate(result_task_vectors):
                    for tvj_id, tv_j in enumerate(result_task_vectors):
                        if tv_id != tvj_id:
                            mask = masks[tv_id][tvj_id]

                            original_norm = torch.norm(tv_j.vector[name], p=2)
                            masked_vector = tv_j.vector[name] * mask
                            masked_norm = torch.norm(masked_vector, p=2)

                            if masked_norm > 0:
                                tv_j.vector[name] = masked_vector * (original_norm / masked_norm)
                            else:
                                tv_j.vector[name] = masked_vector


    ###################################################################
    ###############  tackle the basis parameters  ##################
    ###################################################################
    if args.ratio_bias != 0:
        # tackle the bias of other layers
        for name, pp in list(model.named_parameters()):
            if 'ln' in name and 'weight' in name:
                continue
            elif name == 'model.visual.class_embedding' \
                    or name == 'model.visual.conv1.weight' \
                    or name == 'model.visual.positional_embedding' \
                    or name == 'model.visual.proj' \
                    or len(result_task_vectors[0].vector[name].shape) == 1 \
                    or 'bias' in name:
                params = []
                for tv in result_task_vectors:
                    params.append(tv.vector[name])

                stacked_params = torch.stack(params)
                abs_params = torch.abs(stacked_params)
                flattened_abs_params = abs_params.view(-1)

                num_elements = flattened_abs_params.numel()
                k = int(num_elements * (1-args.ratio_bias))
                threshold = torch.kthvalue(flattened_abs_params, k).values

                mask = abs_params <= threshold

                # Apply the mask to each task vector
                for i, tv in enumerate(result_task_vectors):
                    tv.vector[name] = tv.vector[name] * mask[i]

    return result_task_vectors




# clip vector i version
def bal_mergingi(args, task_vectors, pretrained_checkpoint, exam_datasets, cur_task_vectors=None):
    if cur_task_vectors is None:
        cur_task_vectors = copy.deepcopy(task_vectors)
    result_task_vectors = copy.deepcopy(task_vectors)

    feat_arr = []

    for t_id, exam_dataset in enumerate(exam_datasets):
        tv = cur_task_vectors[t_id]
        model = tv.apply_to(pretrained_checkpoint, scaling_coef=args.scaling_coef_).to('cuda')

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

        # 定义钩子函数
        def hook_fn(module, input, output):
            # 将输入特征移到 CPU 并累加
            accumulated_features[module].append(input[0].detach().cpu())  # input 是元组，取第一个元素
            accumulated_activations[module].append(output[0].detach().cpu())  # input 是元组，取第一个元素

        hooks = []
        for name, param in model.named_parameters():
            # 获取参数所在的模块路径
            module_name = ".".join(name.split(".")[:-1])  # 排除参数本身名称
            if module_name:  # 确保有模块
                # 获取模块对象
                module = dict(model.named_modules())[module_name]
                # 注册钩子
                hook = module.register_forward_hook(hook_fn)
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

        # remove hooks
        for hook in hooks:
            hook.remove()

        feat = {}
        model.to('cpu')
        for name, param in model.named_parameters():

            module_name = ".".join(name.split(".")[:-1])
            if module_name:
                module = dict(model.named_modules())[module_name]
                input_features = accumulated_features.get(module)
                output_features = accumulated_activations.get(module)
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


                elif name == 'model.visual.class_embedding' \
                        or name == 'model.visual.conv1.weight' \
                        or name == 'model.visual.positional_embedding'\
                        or name == 'model.visual.proj'\
                        or len(tv.vector[name].shape) == 1\
                        or 'bias' in name:
                    # tv.vector[name] *= 0.0
                    continue
                else:

                    input_feat_flat = torch.cat(input_features, dim=0)
                    input_feat_flat = input_feat_flat.view(-1, input_feat_flat.shape[-1])  # bs * f_in

                    feat[name] = input_feat_flat
        feat_arr.append(feat)

    # calculate the basis for each task
    basis_arr = []
    for tv_id, tv in enumerate(result_task_vectors):
        basis = {}
        for name, pp in list(model.named_parameters()):
            if tv.vector[name] is None:
                continue
            elif name == 'model.visual.class_embedding' \
                    or name == 'model.visual.conv1.weight' \
                    or name == 'model.visual.positional_embedding' \
                    or name == 'model.visual.proj' \
                    or len(tv.vector[name].shape) == 1 \
                    or 'bias' in name:
                continue
            else:
                feat_i = feat_arr[tv_id][name]    # bs f_in
                if feat_i is None:
                    basis[name] = None
                    continue

                xtx_i = torch.t(feat_i).mm(feat_i)  # f_in f_in
                xtx = xtx_i
                for tvj_id, tv_j in enumerate(result_task_vectors):
                    if tvj_id != tv_id:
                        feat_j = feat_arr[tvj_id][name]
                        xtx -= args.lambd * torch.t(feat_j).mm(feat_j)   # f_in f_in

                w_i = tv.vector[name]     # f_out f_in
                G = w_i.mm(xtx).mm(torch.t(w_i))   # f_out f_out

                # 强制对称化（防止数值误差）
                G = (G + G.T) / 2
                eigenvalues, eigenvectors = torch.linalg.eigh(G)

                if (eigenvalues <= 0).all():
                    basis_ = None
                else:
                    positive_indices = (eigenvalues > 0)
                    positive_eigenvalues = eigenvalues[positive_indices]
                    positive_eigenvectors = eigenvectors[:, positive_indices]

                    if args.ratio == -1.0:  # pick positive basis
                        basis_ = positive_eigenvectors
                    elif args.ratio == 1 or args.ratio > 1:
                        r = eigenvectors.shape[1] - args.ratio

                        basis_ = eigenvectors[:, 0:r]
                    else:
                        val_total = (positive_eigenvalues ** 2).sum()
                        val_ratio = (positive_eigenvalues ** 2) / val_total
                        r = np.sum(np.cumsum(val_ratio) < args.ratio)
                        basis_ = eigenvectors[:, 0:r]

                basis[name] = basis_         # f_out r
                # if basis is None:
                #     print(f"Parameter: {name}, basis None")
                # else:
                #     print(f"Parameter: {name}, basis shape: {basis_.shape}")

        basis_arr.append(basis)

    for tv_id, tv in enumerate(result_task_vectors):
        basis = basis_arr[tv_id]
        for name, pp in list(model.named_parameters()):
            if tv.vector[name] is None:
                continue
            elif name == 'model.visual.class_embedding' \
                    or name == 'model.visual.conv1.weight' \
                    or name == 'model.visual.positional_embedding'\
                    or name == 'model.visual.proj'\
                    or len(tv.vector[name].shape) == 1\
                    or 'bias' in name:
                continue
            else:
                b = basis[name]  # f_out r
                if b is None:
                    continue
                bbt_ = b.mm(torch.t(b))  # f_out f_out

                vector_i = tv.vector[name] # f_out f_in

                vector_orthogonal = torch.mm(bbt_, vector_i)

                norm_original = vector_i.norm()
                norm_remaining = vector_orthogonal.norm()
                tv.vector[name] = vector_orthogonal * (norm_original / norm_remaining)


    return result_task_vectors











