import time
import numpy as np
import torch
import os
from src.task_vectors_linear import  LinearizedTaskVector
from eval import eval_single_dataset
from args import parse_arguments


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
    # Config
    models = ['ViT-B-16']  # , 'ViT-L-14''ViT-B-32',
    datasets = [
        "Cars",
        "DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SUN397",
        "SVHN",
    ]
    # model = models[2]
    for model in models:
        args = parse_arguments()
        args.finetuning_mode = "linear"
        args.data_location = '\\Chenzebin\\data'
        args.model = model
        args.save = f'checkpoints/{model}'
        args.logs_path = 'logs/' + model
        pretrained_checkpoint = f'checkpoints/{model}/DTDVal/linear_zeroshot.pt'

        # 创建log
        str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        log = create_log_dir(args.logs_path, 'log_{}_Linear_TA_merge.txt'.format(str_time_))
        args.log = log

        # Create the task vectors
        task_vectors = [
            LinearizedTaskVector(pretrained_checkpoint, f'checkpoints/{model}/{dataset}Val/linear_finetuned.pt')
            for dataset in datasets
        ]
        # Sum the task vectors
        task_vector_sum = sum(task_vectors)

        best_scaling_coef = 0

        best_total_metrics = 0

        for scaling_coef in np.arange(0, 0.55, 0.05):
            print(f"此时的系数为:{scaling_coef}")
            # Apply the resulting task vector
            image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef)
            # Evaluate

            total_metrics = 0
            dataset_num = 0
            for dataset in datasets:
                metrics = eval_single_dataset(image_encoder, dataset, args)

                total_metrics += metrics['top1']
                log.info(' dataset: ' + str(dataset) + ' ACC: ' + str(metrics['top1']))
                dataset_num += 1

            log.info('scaling_coef:' + str(scaling_coef))

            log.info(' avg_metrics:' + str(total_metrics / len(datasets)) + '\n')
            if total_metrics > best_total_metrics:
                best_total_metrics = total_metrics
                best_scaling_coef = scaling_coef

        log.info('best scaling_coef: ' + str(best_scaling_coef))


if __name__ == '__main__':
    main()