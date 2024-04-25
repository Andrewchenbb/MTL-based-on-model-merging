import os
import time

import torch

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset# 从注册表中获取指定的数据集
#from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.eval import eval_single_dataset
from src.heads import get_classification_head

from src.linearize import LinearizedImageEncoder
from src.modeling import ImageClassifier, ImageEncoder
from src.utils import LabelSmoothing, cosine_lr


def finetune(args):
    # 设置分布式数据并行环境
    #setup_ddp(rank, args.world_size, port=args.port)

    # 准备模型保存路径和断言微调模式有效性
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)

    # 断言微调模式必须是"linear"或"standard"之一
    assert args.finetuning_mode in [##################################
        "linear",
        "standard",
    ], "Only linear and standard fine-tuning are supported."

    # 判断是否使用线性化微调
    linearized_finetuning = args.finetuning_mode == "linear"####################
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    # 检查是否已经存在模型检查点，如果存在，则跳过微调
    # Check if checkpoints already exist
    ft_path = (##################
        os.path.join(args.save, train_dataset, "linear_finetuned.pt")
        if linearized_finetuning
        else os.path.join(args.save, train_dataset, "finetuned.pt")
    )
    zs_path = (
        os.path.join(args.save, train_dataset, "linear_zeroshot.pt")
        if linearized_finetuning
        else os.path.join(args.save, train_dataset, "zeroshot.pt")
    )
    if os.path.exists(zs_path) and os.path.exists(ft_path):
        print(f"Skipping fine-tuning because {ft_path} exists.")
        return zs_path, ft_path

    # 确保指定了训练数据集
    assert train_dataset is not None, "Please provide a training dataset."

    # 根据微调模式加载预训练的图像编码器
    if args.load is not None and args.load.endswith("pt"):
        image_encoder = (#########################
            LinearizedImageEncoder.load(args.load)
            if linearized_finetuning
            else ImageEncoder.load(args.load)
        )
    else:
        print("Building image encoder.")
        image_encoder = (
            LinearizedImageEncoder(args, keep_lang=False)
            if linearized_finetuning
            else ImageEncoder(args)
        )

    # 获取分类头并构建图像分类模型
    classification_head = get_classification_head(args, train_dataset)

    # 构建图像分类模型
    model = ImageClassifier(image_encoder, classification_head)

    # 冻结模型头部，准备模型和数据加载器
    model.freeze_head()
    model = model.cuda()###############

    # 获取数据预处理函数和打印间隔
    preprocess_fn = model.train_preprocess
    print_every = 100

    # 加载数据集和数据加载器
    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)###########
    num_batches = len(dataset.train_loader)

    # 分布式数据加载和模型并行化
    # Distribute the data and model across the GPUs.
    # ddp_loader = distribute_loader(data_loader)
    # ddp_model = torch.nn.parallel.DistributedDataParallel(
    #     model,
    #     device_ids=[rank],
    #     find_unused_parameters=True,
    #     output_device=rank,
    # )

    # 设置损失函数，支持标签平滑
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    # 设置优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    # 设置学习率调度器
    scheduler = cosine_lr(
        optimizer,
        args.lr,
        args.warmup_length,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    # # 如果是主进程且指定了保存路径，则保存零次微调（zero-shot）模型
    # Saving zero-shot model
    if args.save is not None :
        os.makedirs(ckpdir, exist_ok=True)
        model_path = (#######################
            os.path.join(ckpdir, "linear_zeroshot.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, "zeroshot.pt")
        )
        model.image_encoder.save(model_path)############名字

    # 训练循环：按照指定的周期数进行训练
    for epoch in range(args.epochs):
        model.train()#########名字

        for i, batch in enumerate(data_loader):#####名字
            start_time = time.time()

            # 计算当前步数
            step = (################################
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            # 将批次数据转换为字典格式，并移动到GPU上
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            labels = batch["labels"].cuda()
            data_time = time.time() - start_time

            logits = model(inputs)

            loss = loss_fn(logits, labels)

            loss.backward()

            # 如果达到梯度累积步数，则更新模型参数###############
            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            # 定期保存检查点和打印训练信息
            if (
                args.checkpoint_every > 0###########
                and step % args.checkpoint_every == 0
            ):
                print("Saving checkpoint.")
                model_path = (
                    os.path.join(ckpdir, f"linear_checkpoint_{step}.pt")
                    if linearized_finetuning
                    else os.path.join(ckpdir, f"checkpoint_{step}.pt")
                )
                model.image_encoder.save(model_path)

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)

            ):
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"  # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                    flush=True,
                )




    image_encoder = model.image_encoder
    eval_single_dataset(image_encoder, train_dataset, args)

    # 如果是主进程且指定了保存路径，则保存微调后的模型
    if args.save is not None:
        zs_path = (
            os.path.join(ckpdir, "linear_zeroshot.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, "zeroshot.pt")
        )
        ft_path = (
            os.path.join(ckpdir, "linear_finetuned.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, "finetuned.pt")
        )
        image_encoder.save(ft_path)
        return zs_path, ft_path
    # 清理分布式数据并行环境



if __name__ == "__main__":
    data_location = '\\Chenzebin\\data'
    models = ['ViT-L-14']#,'ViT-B-32', 'ViT-B-16'
    train_datasets = [
        "Cars",
        "DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SUN397",
        "SVHN",
    ]
    epochs = {
        "Cars": 35,
        "DTD": 76,
        "EuroSAT": 12,
        "GTSRB": 11,
        "MNIST": 5,
        "RESISC45": 15,
        "SUN397": 14,
        "SVHN": 4,
    }

    for model in models:
        for dataset in train_datasets:
            args = parse_arguments()

            # HACK: Some command line arguments are overwritten by defaults here.

            args.lr = 1e-5
            args.epochs = epochs[dataset]
            args.data_location = data_location
            args.train_dataset = dataset + "Val"

            # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
            args.batch_size = 8 if args.model == "ViT-L-14" else 128
            args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
            args.model = model
            args.finetuning_mode="linear"


            if args.seed is not None:
                args.save = f"checkpoints_{args.seed}/{args.model}"
            else:
                args.save = f"checkpoints/{args.model}"
            print("=" * 100)
            print(f"Finetuning {args.model} on {dataset}")
            print("=" * 100)
            finetune(args)
