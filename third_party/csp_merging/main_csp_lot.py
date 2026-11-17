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

from methods.lot_merging_utils import lot_merging
from methods.rotate_ta import rotate_filter_pertv


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


exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']  # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD
test_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']  # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD

args = parse_arguments()


model = 'ViT-B-32'
args.scaling_coef_ = 1.4
args.repeat = 3
args.exp_size = 64
args.parts = ['outout']
args.k = 5

# model = 'ViT-L-14'
# args.scaling_coef_ = 1.6
# args.repeat = 3
# args.exp_size = 64
# args.batch_size = 16
# args.parts = ['outout']
# args.k = 20


args.data_location = '../data'
args.model = model
args.device = 'cuda'
args.save = '../checkpoints/checkpoints/' + model
args.logs_path = '../logs/' + model
pretrained_checkpoint = '../checkpoints/checkpoints/' + model + '/zeroshot.pt'

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_task_arithmetic.txt'.format(str_time_))
for conf in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
    args.k = conf

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

        ################################################################
        ######################### Testing ##############################
        ################################################################
        start = time.time()
        opt_vector = lot_merging(args, task_vectors, pretrained_checkpoint, exam_datasets)

        rotate_filter_pertv(args, opt_vector, pretrained_checkpoint)

        image_encoder = opt_vector.apply_to(pretrained_checkpoint, scaling_coef=args.scaling_coef_)
        log.info('*'*20 + 'scaling_coef:' + str(args.scaling_coef_) + '*'*20)

        end = time.time()
        print(f"Time for merging: {end - start:.4f} seconds")

        accs = []
        for dataset in test_datasets:
            metrics = eval_single_dataset(image_encoder, dataset, args)
            log.info(str(dataset) + ':' + str(metrics.get('top1')*100)+'%')
            accs.append(metrics.get('top1')*100)
        log.info('Avg ACC:' + str(np.mean(accs)) + '%')

        acc_all.append(accs)


    acc_all = np.array(acc_all)
    dataset_means = np.mean(acc_all, axis=0)
    dataset_stds = np.std(acc_all, axis=0)

    print("\n############# ACC for each dataset #############")
    for i, dataset in enumerate(test_datasets):
        print(f"{dataset}: AVG={dataset_means[i]:.2f}%, STD={dataset_stds[i]:.2f}%")

    overall_means = np.mean(acc_all, axis=1)
    overall_std = np.std(overall_means)
    overall_mean = np.mean(overall_means)

    print("\n####################################################")
    print(f"Average performance across all datasets: AVG={overall_mean:.2f}%, STD={overall_std:.2f}%")





