import torch
import copy
import torch.nn.functional as F

from datasets.common import get_dataloader
from datasets.registry import get_dataset
from functools import partial

from collections import defaultdict
import numpy as np



# 将对W_pre旋转的部分去掉再进行Merge
def rotate_ta(args, task_vectors, pretrained_checkpoint):
    opt_tv = copy.deepcopy(sum(task_vectors))
    model = torch.load(pretrained_checkpoint).to('cuda')
    for name, pp in list(model.named_parameters()):
        if opt_tv.vector[name] is None:
            continue
        elif len(pp.shape) == 1:
            pass
            # tvs = [tv.vector[name] for tv in task_vectors]
            # T_optimal = _cal_optimal_tv_1d(tvs)
            # opt_tv.vector[name] = T_optimal
        elif len(pp.shape) == 2:
            tvs = [tv.vector[name].to('cuda') for tv in task_vectors]
            W_pre = pp.data.to('cuda')

            U, S, Vh = torch.linalg.svd(W_pre, full_matrices=False)  # U:[d_out,m], S:[m], Vh:[m,d_in]
            m = S.shape[0]
            if args.k == -1:
                k = m
            else:
                k = args.k
            r = min(k, m)
            U_r = U[:, :r].contiguous()
            S_r = S[:r].contiguous()
            V_r = Vh[:r, :].transpose(-2, -1).contiguous()

            T_optimal = 0.0
            for tv in tvs:
                # P_U W  = U(U^T W)
                PUW = U_r @ (U_r.transpose(-2, -1) @ tv)  # [d_out, d_in]
                # W P_V  = (W V) V^T
                WPV = (tv @ V_r) @ V_r.transpose(-2, -1)  # [d_out, d_in]
                # P_U W P_V
                W_in_in = U_r @ (U_r.transpose(-2, -1) @ tv @ V_r) @ V_r.transpose(-2, -1)
                # out U only, in V
                # W_outU_inV = WPV - W_in_in  # (I-P_U) W P_V
                # in U, out V only
                # W_inU_outV = PUW - W_in_in  # P_U W (I-P_V)
                # out both: (I-P_U) W (I-P_V) = W - P_UW - WPV + P_UWPV
                W_out_out = tv - PUW - WPV + W_in_in

                T_optimal += W_in_in
                if args.include_r:
                    T_optimal += W_out_out

            opt_tv.vector[name] = T_optimal.to('cpu')

    for name, pp in list(model.named_parameters()):
        opt_tv.vector[name] = opt_tv.vector[name] / len(task_vectors)

    return opt_tv

# 不 merge，仅对task vector处理，方便集成到其他方法
def rotate_ta_filter(args, task_vectors, pretrained_checkpoint):
    opt_tv = copy.deepcopy(sum(task_vectors))
    model = torch.load(pretrained_checkpoint).to('cuda')
    for name, pp in list(model.named_parameters()):
        if opt_tv.vector[name] is None:
            continue
        elif len(pp.shape) == 1:
            pass
            # tvs = [tv.vector[name] for tv in task_vectors]
            # T_optimal = _cal_optimal_tv_1d(tvs)
            # opt_tv.vector[name] = T_optimal
        elif len(pp.shape) == 2:
            W_pre = pp.data.to('cuda')

            U, S, Vh = torch.linalg.svd(W_pre, full_matrices=False)  # U:[d_out,m], S:[m], Vh:[m,d_in]
            m = S.shape[0]
            if args.k == -1:
                k = m
            else:
                k = args.k
            r = min(k, m)
            U_r = U[:, :r].contiguous()
            S_r = S[:r].contiguous()
            V_r = Vh[:r, :].transpose(-2, -1).contiguous()

            for task_vector in task_vectors:
                tv = task_vector.vector[name].to('cuda')

                tv_filtered = 0.0

                # P_U W  = U(U^T W)
                PUW = U_r @ (U_r.transpose(-2, -1) @ tv)  # [d_out, d_in]
                # W P_V  = (W V) V^T
                WPV = (tv @ V_r) @ V_r.transpose(-2, -1)  # [d_out, d_in]
                # P_U W P_V
                W_in_in = U_r @ (U_r.transpose(-2, -1) @ tv @ V_r) @ V_r.transpose(-2, -1)
                # out U only, in V
                # W_outU_inV = WPV - W_in_in  # (I-P_U) W P_V
                # in U, out V only
                # W_inU_outV = PUW - W_in_in  # P_U W (I-P_V)
                # out both: (I-P_U) W (I-P_V) = W - P_UW - WPV + P_UWPV
                W_out_out = tv - PUW - WPV + W_in_in

                tv_filtered += W_in_in
                if args.include_r:
                    tv_filtered += W_out_out
                task_vector.vector[name] = tv_filtered.to('cpu')


# 处理单个task vector 目前使用这个，方便集成到已有方法中
def rotate_filter_pertv(args, task_vector, pretrained_checkpoint):
    model = torch.load(pretrained_checkpoint).to('cuda')
    for name, pp in list(model.named_parameters()):
        if task_vector.vector[name] is None:
            continue
        elif len(pp.shape) == 1:
            pass
        elif len(pp.shape) == 2:
            # SVD on pretrained vector
            W_pre = pp.data.to('cuda')

            U, S, Vh = torch.linalg.svd(W_pre, full_matrices=False)  # U:[d_out,m], S:[m], Vh:[m,d_in]
            m = S.shape[0]
            if args.k == -1:
                k = m
            elif args.k < 1:
                energy = S.pow(2)
                cum_energy = torch.cumsum(energy, dim=0)
                total_energy = cum_energy[-1].clamp_min(1e-12)  # 防止全0
                target = args.k * total_energy

                k = int(torch.searchsorted(cum_energy, target).item()) + 1
                print('top ', args.k, 'singular is k=', k)
            else:
                k = args.k
            r = min(k, m)
            U_r = U[:, :r].contiguous()
            S_r = S[:r].contiguous()
            V_r = Vh[:r, :].transpose(-2, -1).contiguous()


            # Decomposing TV into 4 parts
            tv = task_vector.vector[name].to('cuda')

            # P_U W  = U(U^T W)
            PUW = U_r @ (U_r.transpose(-2, -1) @ tv)  # [d_out, d_in]
            # W P_V  = (W V) V^T
            WPV = (tv @ V_r) @ V_r.transpose(-2, -1)  # [d_out, d_in]
            # P_U W P_V
            W_in_in = U_r @ (U_r.transpose(-2, -1) @ tv @ V_r) @ V_r.transpose(-2, -1)
            # out U only, in V
            W_outU_inV = WPV - W_in_in  # (I-P_U) W P_V
            # in U, out V only
            W_inU_outV = PUW - W_in_in  # P_U W (I-P_V)
            # out both: (I-P_U) W (I-P_V) = W - P_UW - WPV + P_UWPV
            W_out_out = tv - PUW - WPV + W_in_in


            # Selecting the merging parts of TV
            tv_filtered = 0.0
            if 'inin' in args.parts:
                tv_filtered += W_in_in
            if 'inout' in args.parts:
                tv_filtered += W_inU_outV
            if 'outin' in args.parts:
                tv_filtered += W_outU_inV
            if 'outout' in args.parts:
                tv_filtered += W_out_out

            task_vector.vector[name] = tv_filtered.to('cpu')


# ablation实验，包括bottom_k 和 random 随机子空间
def rotate_filter_pertv_ablation(args, task_vector, pretrained_checkpoint, type='bottom'):
    model = torch.load(pretrained_checkpoint).to('cuda')
    for name, pp in list(model.named_parameters()):
        if task_vector.vector[name] is None:
            continue
        elif len(pp.shape) == 1:
            pass
        elif len(pp.shape) == 2:
            # SVD on pretrained vector
            W_pre = pp.data.to('cuda')

            if type == 'bottom':
                U, S, Vh = torch.linalg.svd(W_pre, full_matrices=False)  # U:[d_out,m], S:[m], Vh:[m,d_in]
                m = S.shape[0]
                if args.k == -1:
                    k = m
                elif args.k < 1:
                    energy = S.pow(2)
                    cum_energy = torch.cumsum(energy, dim=0)
                    total_energy = cum_energy[-1].clamp_min(1e-12)  # 防止全0
                    target = args.k * total_energy

                    k = int(torch.searchsorted(cum_energy, target).item()) + 1
                    print('top ', args.k, 'singular is k=', k)
                else:
                    k = args.k
                r = min(k, m)
                U_r = U[:, -r:].contiguous()
                S_r = S[-r:].contiguous()
                V_r = Vh[-r:, :].transpose(-2, -1).contiguous()
            elif type == 'random':
                w_rand = torch.rand_like(W_pre)
                U, S, Vh = torch.linalg.svd(w_rand, full_matrices=False)  # U:[d_out,m], S:[m], Vh:[m,d_in]
                m = S.shape[0]
                if args.k == -1:
                    k = m
                elif args.k < 1:
                    energy = S.pow(2)
                    cum_energy = torch.cumsum(energy, dim=0)
                    total_energy = cum_energy[-1].clamp_min(1e-12)  # 防止全0
                    target = args.k * total_energy

                    k = int(torch.searchsorted(cum_energy, target).item()) + 1
                    print('top ', args.k, 'singular is k=', k)
                else:
                    k = args.k
                r = min(k, m)
                U_r = U[:, -r:].contiguous()
                S_r = S[-r:].contiguous()
                V_r = Vh[-r:, :].transpose(-2, -1).contiguous()
            else:
                print('type 错了')
                exit()


            # Decomposing TV into 4 parts
            tv = task_vector.vector[name].to('cuda')

            # P_U W  = U(U^T W)
            PUW = U_r @ (U_r.transpose(-2, -1) @ tv)  # [d_out, d_in]
            # W P_V  = (W V) V^T
            WPV = (tv @ V_r) @ V_r.transpose(-2, -1)  # [d_out, d_in]
            # P_U W P_V
            W_in_in = U_r @ (U_r.transpose(-2, -1) @ tv @ V_r) @ V_r.transpose(-2, -1)
            # out U only, in V
            W_outU_inV = WPV - W_in_in  # (I-P_U) W P_V
            # in U, out V only
            W_inU_outV = PUW - W_in_in  # P_U W (I-P_V)
            # out both: (I-P_U) W (I-P_V) = W - P_UW - WPV + P_UWPV
            W_out_out = tv - PUW - WPV + W_in_in



            # Selecting the merging parts of TV
            tv_filtered = 0.0
            if 'inin' in args.parts:
                tv_filtered += W_in_in
            if 'inout' in args.parts:
                tv_filtered += W_inU_outV
            if 'outin' in args.parts:
                tv_filtered += W_outU_inV
            if 'outout' in args.parts:
                tv_filtered += W_out_out

            task_vector.vector[name] = tv_filtered.to('cpu')



# 处理单个task vector 目前使用这个，方便集成到已有方法中
def rotate_filter_pertv_p(args, task_vector, pretrained_checkpoint):
    model = torch.load(pretrained_checkpoint).to('cuda')
    for name, pp in list(model.named_parameters()):
        if task_vector.vector[name] is None:
            continue
        elif len(pp.shape) == 1:
            pass
        elif len(pp.shape) == 2:
            # SVD on pretrained vector
            W_pre = pp.data.to('cuda')

            U, S, Vh = torch.linalg.svd(W_pre, full_matrices=False)  # U:[d_out,m], S:[m], Vh:[m,d_in]
            m = S.shape[0]

            # 首先选取 top p
            energy = S.pow(2)
            cum_energy = torch.cumsum(energy, dim=0)
            total_energy = cum_energy[-1].clamp_min(1e-12)  # 防止全0
            target = args.p * total_energy

            k = int(torch.searchsorted(cum_energy, target).item()) + 1
                # print('top ', args.p, 'singular is k=', k)
            if k > args.k:
                k = args.k

            r = min(k, m)
            U_r = U[:, :r].contiguous()
            S_r = S[:r].contiguous()
            V_r = Vh[:r, :].transpose(-2, -1).contiguous()


            # Decomposing TV into 4 parts
            tv = task_vector.vector[name].to('cuda')

            # P_U W  = U(U^T W)
            PUW = U_r @ (U_r.transpose(-2, -1) @ tv)  # [d_out, d_in]
            # W P_V  = (W V) V^T
            WPV = (tv @ V_r) @ V_r.transpose(-2, -1)  # [d_out, d_in]
            # P_U W P_V
            W_in_in = U_r @ (U_r.transpose(-2, -1) @ tv @ V_r) @ V_r.transpose(-2, -1)
            # out U only, in V
            W_outU_inV = WPV - W_in_in  # (I-P_U) W P_V
            # in U, out V only
            W_inU_outV = PUW - W_in_in  # P_U W (I-P_V)
            # out both: (I-P_U) W (I-P_V) = W - P_UW - WPV + P_UWPV
            W_out_out = tv - PUW - WPV + W_in_in




            # Selecting the merging parts of TV
            tv_filtered = 0.0
            if 'inin' in args.parts:
                tv_filtered += W_in_in
            if 'inout' in args.parts:
                tv_filtered += W_inU_outV
            if 'outin' in args.parts:
                tv_filtered += W_outU_inV
            if 'outout' in args.parts:
                tv_filtered += W_out_out

            task_vector.vector[name] = tv_filtered.to('cpu')



































