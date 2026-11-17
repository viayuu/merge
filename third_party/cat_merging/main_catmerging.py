import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import sys
sys.path.append('../')

import torch
import numpy as np

from utils.task_vectors import TaskVector
from utils.eval import eval_single_dataset
from utils.args import parse_arguments

from methods.cat_merging_utils import bal_mergingj


def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


#exam_datasets = ['SUN397', 'Cars']  # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD
exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']  # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD
test_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST','DTD']  # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD

args = parse_arguments()
args.is_imageNet = False
args.scaling_coef_ = 0.4
args.repeat = 3

model = 'ViT-B-32'
args.exp_size = 2
args.ratio = 4
args.lambd = 5
args.ratio_ln = 4
args.lambd_ln = 10
args.ratio_bias = 0.25

# model = 'ViT-B-16'
# args.exp_size = 3
# args.ratio = 4
# args.lambd = 50
# args.ratio_ln = 4
# args.lambd_ln = 50
# args.ratio_bias = 0.25

# model = 'ViT-L-14'
# args.exp_size = 3
# args.batch_size = 16
# args.ratio = 2
# args.lambd = 50
# args.ratio_ln = 2
# args.lambd_ln = 50
# args.ratio_bias = 0.15


args.data_location = '../data'
args.model = model
args.device = 'cuda'
args.save = '../checkpoints/checkpoints/' + model
args.logs_path = '../logs/' + model
pretrained_checkpoint = '../checkpoints/checkpoints/' + model + '/zeroshot.pt'

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_task_arithmetic.txt'.format(str_time_))
for conf in [0]:

    args.ratio_ln = 0
    args.ratio_bias = 0
    test_datasets = []
    print('################################################################')
    print('######################### Merging :', conf, ' ##############################')
    print('################################################################')
    print(args)

    acc_all = []
    for run in range(args.repeat):
        print('######################### Run :', run, ' ##############################')

        task_vectors = [
            TaskVector(pretrained_checkpoint, '../checkpoints/checkpoints/' + model + '/' + dataset_name + '/finetuned.pt')
            for dataset_name in exam_datasets
        ]
        pretrained_model = torch.load(pretrained_checkpoint).to('cuda')

        start = time.time()
        task_vectors = bal_mergingj(args, task_vectors, pretrained_checkpoint, exam_datasets)
        end = time.time()
        print(f"代码执行时间：{end - start:.4f} 秒")

        ################################################################
        ######################### Testing ##############################
        ################################################################

        task_vector_sum = sum(task_vectors)
        image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=args.scaling_coef_)
        log.info('*'*20 + 'scaling_coef:' + str(args.scaling_coef_) + '*'*20)

        accs = []
        for dataset in test_datasets:
            metrics = eval_single_dataset(image_encoder, dataset, args)
            log.info(str(dataset) + ':' + str(metrics.get('top1')*100)+'%')
            accs.append(metrics.get('top1')*100)
        log.info('Avg ACC:' + str(np.mean(accs)) + '%')

        acc_all.append(accs)

    # ================= 统计部分 ====================
    # 转换 acc_all 为 numpy 数组以便计算
    acc_all = np.array(acc_all)  # 维度: [repeat, len(exam_datasets)]

    # 1. 每个数据集的平均值和方差
    dataset_means = np.mean(acc_all, axis=0)  # 每列的平均值
    dataset_stds = np.std(acc_all, axis=0)  # 每列的方差

    print("\n############# ACC for each dataset #############")
    for i, dataset in enumerate(exam_datasets):
        print(f"{dataset}: AVG={dataset_means[i]:.2f}%, STD={dataset_stds[i]:.2f}%")

    # 2. 所有数据集平均准确率的5次运行平均方差
    overall_means = np.mean(acc_all, axis=1)  # 每次运行中所有数据集的平均准确率
    overall_std = np.std(overall_means)  # 5次运行平均准确率的方差
    overall_mean = np.mean(overall_means)  # 所有运行的总体平均值

    print("\n############# 总体统计结果 #############")
    print(f"Average performance across all datasets: AVG={overall_mean:.2f}%, STD={overall_std:.2f}%")






