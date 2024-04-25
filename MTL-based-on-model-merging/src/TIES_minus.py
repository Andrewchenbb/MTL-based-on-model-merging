import time

from task_vectors import TaskVector
from eval import eval_single_dataset
from args import parse_arguments
import numpy as np
import os

from TIES_utils import *

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
    fh = logging.FileHandler(path+'/'+filename)
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
    models = [ 'ViT-L-14' ]#'ViT-B-32','ViT-B-16'

    datasets = [
        #"Cars",
        #"DTD",
        #"EuroSAT",
        #"GTSRB",
        "MNIST",
        #"RESISC45",
        #"SUN397",
        #"SVHN",
    ]
    args = parse_arguments()
    args.data_location = '\\Chenzebin\\data'


    for model in models:
        #设定控制任务精度，减少训练步骤
        if model=='ViT-B-32':
            metrics_pre=0.6335
        elif model=='ViT-B-16':
            metrics_pre=0.68334
        elif model=='ViT-L-14':
            metrics_pre=0.75542


        args.logs_path = 'logs/' + model
        args.model = model
        args.save = f'checkpoints/{model}'
        # 创建log
        str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        log = create_log_dir(args.logs_path, 'log_{}_TIES_minus.txt'.format(str_time_))
        args.log = log

        for dataset in datasets:
            # Config
            pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'
            # finetuned_checkpoint = f'checkpoints/{model}/{dataset}/finetuned.pt'
            ft_checks=[torch.load('checkpoints/' + model + '/' + dataset_name + '/finetuned.pt').state_dict()
             for dataset_name in [dataset]]
            pt_check=torch.load(pretrained_checkpoint).state_dict()
            #检查是否参数名称是否一致
            check_parameterNamesMatch(ft_checks + [pt_check])

            remove_keys = []
            print(f"Flattening out Checkpoints")
            # 将状态字典扁平化为向量
            flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
            flat_ptm = state_dict_to_vector(pt_check, remove_keys)

            tv_flat_checks = flat_ft - flat_ptm  # 计算任务向量和预训练模型向量的差异
            assert check_state_dicts_equal(vector_to_state_dict(flat_ptm, pt_check, remove_keys), pt_check)
            assert all(
                [check_state_dicts_equal(vector_to_state_dict(flat_ft[i], pt_check, remove_keys), ft_checks[i]) for i
                 in range(len(ft_checks))])

            K = 20
            merge_func = "dis-sum"


            # 使用ties_merging函数获得裁剪、选择、不相交合并的任务向量
            merged_tv = ties_merging(tv_flat_checks, reset_thresh=K, merge_func=merge_func, )


            best_sca=0
            best_metrics=1.0
            for scaling_coef in np.arange(0.6, 0.65, 0.05):
                print(f"此时的系数为:{scaling_coef}")
                merged_check = flat_ptm - scaling_coef * merged_tv  # 将合并后的任务向量按比例加回到预训练模型向量中
                merged_state_dict = vector_to_state_dict(merged_check, pt_check, remove_keys=remove_keys)

                image_encoder = torch.load(pretrained_checkpoint)
                image_encoder.load_state_dict(merged_state_dict, strict=False)  # 加载合并后的状态字典到模型
                # Evaluate
                metrics_task=eval_single_dataset(image_encoder, dataset, args)#4.86
                metrics_imagenet=eval_single_dataset(image_encoder, 'ImageNet', args)
                #eval_single_dataset(image_encoder1, dataset, args)#5.27
                # 记录原始模型以及微调模型在目标任务及控制任务的精度,当lambda=0时，相当于原始模型

                if metrics_task['top1']<best_metrics and metrics_imagenet['top1']>0.95*metrics_pre:
                    best_metrics=metrics_task['top1']
                    best_sca=scaling_coef
                    metrics_control=metrics_imagenet['top1']
                log.info('scaling_coef: '+str(scaling_coef)+' task: '+str(dataset)+' metrics: '+str(metrics_task['top1'])+'metrics_control: '+str(metrics_imagenet['top1']))

            log.info('TIES_best_scaling_coef:'+str(best_sca)+' ACC:'+str(best_metrics)+' control_task ACC:'+str(metrics_control)+ '\n')




if __name__ == '__main__':
    main()