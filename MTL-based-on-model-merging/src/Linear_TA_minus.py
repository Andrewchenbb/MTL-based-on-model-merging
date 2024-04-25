import time

import torch
#from task_vectors import TaskVector
from src.task_vectors_linear import  LinearizedTaskVector
from eval import eval_single_dataset
from args import parse_arguments
import numpy as np
import os


def create_log_dir(path, filename='log.txt'):
    # 定义一个函数，用于创建日志目录和配置日志记录器。
    # 导入logging模块，用于记录日志。
    import logging
    # 检查指定的路径是否存在。
    if not os.path.exists(path):
        os.makedirs(path)
    # 创建一个日志记录器。
    logger = logging.getLogger(path)
    # 设置日志记录器的级别为DEBUG。
    logger.setLevel(logging.DEBUG)
    # 创建一个文件日志处理器，用于将日志记录到文件中。
    fh = logging.FileHandler(path + '/' + filename)
    # 设置文件日志处理器的级别为DEBUG。
    fh.setLevel(logging.DEBUG)
    # 创建一个流日志处理器，用于将日志输出到控制台。
    ch = logging.StreamHandler()
    # 设置流日志处理器的级别为DEBUG。
    ch.setLevel(logging.DEBUG)
    # 将文件日志处理器添加到日志记录器。
    logger.addHandler(fh)
    # 将流日志处理器添加到日志记录器。
    logger.addHandler(ch)
    return logger


def main():
    models = ['ViT-B-16']#'ViT-L-14''ViT-B-32',

    datasets = [
        #"Cars",
        #"DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SUN397",
        "SVHN",
    ]
    args = parse_arguments()
    args.finetuning_mode = "linear"
    args.data_location = '\\Chenzebin\\data'

    for model in models:
        args.logs_path = 'logs/' + model
        args.model = model
        args.save = f'checkpoints/{model}'
        # 创建log
        str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        log = create_log_dir(args.logs_path, 'log_{}_Linear_TA_minus.txt'.format(str_time_))
        args.log = log

        for dataset in datasets:
            # Config
            linear_zeroshot_checkpoint = f'checkpoints/{model}/{dataset}Val/linear_zeroshot.pt'
            linear_finetuned_checkpoint = f'checkpoints/{model}/{dataset}Val/linear_finetuned.pt'

            # Create the task vector
            linear_task_vector = LinearizedTaskVector(linear_zeroshot_checkpoint, linear_finetuned_checkpoint)
            # Negate the task vector
            neg_linear_task_vector = -linear_task_vector

            best_sca = 0
            best_metrics = 1

            #记录预训练
            image_encoder = neg_linear_task_vector.apply_to(linear_zeroshot_checkpoint, 0)
            # Evaluate
            metrics_task = eval_single_dataset(image_encoder, dataset, args)
            log.info('linear_zeroshot_model ' + ' dataset: ' + str(dataset) + ' ACC: ' + str(metrics_task['top1']))
            metrics_pre = eval_single_dataset(image_encoder, 'ImageNet', args)
            log.info('linear_zeroshot_model in control_task ' + ' ACC: ' + str(metrics_pre['top1']))



            for scaling_coef in np.arange(0.5, 1.05, 0.05):
                print(f"此时的系数为:{scaling_coef}")
                image_encoder = neg_linear_task_vector.apply_to(linear_zeroshot_checkpoint, scaling_coef)

                # Evaluate
                metrics_task = eval_single_dataset(image_encoder, dataset, args)  # 4.86

                # 记录原始模型以及微调模型在目标任务及控制任务的精度,当lambda=0时，相当于原始模型
                # if scaling_coef == 0:
                #     log.info('pretrain_model ' + ' dataset: ' + str(dataset) + ' ACC: ' + str(metrics_minst['top1']))
                #     metrics_pre = eval_single_dataset(image_encoder, 'ImageNet', args)
                #     log.info('pretrain_model in control_task ' + ' ACC: ' + str(metrics_pre['top1']))
                if metrics_task['top1'] < best_metrics:
                    metrics_image=eval_single_dataset(image_encoder, 'ImageNet', args)
                    log.info('scaling_coef: '+str(scaling_coef)+"control_ACC: "+str(metrics_image['top1']))
                    if metrics_image['top1']>0.95*metrics_pre['top1']:
                        best_metrics = metrics_task['top1']
                        best_sca = scaling_coef
                        metrics_control=metrics_image['top1']
                log.info('scaling_coef: '+str(scaling_coef)+' task: '+str(dataset)+' ACC: '+str(metrics_task))

            log.info('the best scaling_coef：' + str(best_sca) + ' ACC:' + str(best_metrics) + ' control_task ACC:' + str(
                metrics_control) +'/n')


if __name__ == '__main__':
    main()