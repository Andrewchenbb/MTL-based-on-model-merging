import os

from tqdm import tqdm
import tqdm

from args import parse_arguments
from src.eval import eval_single_dataset,eval_single_dataset_preprocess_head
from src.linearize import LinearizedImageEncoder
from src.task_vectors_linear import  LinearizedTaskVector
import torch
from heads import get_classification_head
import time
from datasets.registry import get_dataset
from datasets.common import get_dataloader, maybe_dictionarize, get_dataloader_shuffle


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

#递归删除一个对象的属性
def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

#递归设置一个对象的属性
def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

#返回模型的所有参数以及参数名称列表，移除可训练参数
def make_functional(mod):
    orig_params = tuple(mod.parameters())
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

#将之前保存的参数重新加载到模型中
def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

class ModelWrapper(torch.nn.Module):
    # 定义一个模型包装器类，继承自torch.nn.Module。
    def __init__(self, model, initial_weights=None):
        # 构造函数接收一个模型实例和初始权重作为可选参数。
        super(ModelWrapper, self).__init__()
        self.model = model

        # 检查模型是否有名为'transformer'的属性。
        if hasattr(self.model, 'transformer'):
            # 如果有，则删除该属性。这可能是为了移除模型中的某些部分，以便于修改或替换。
            delattr(self.model, 'transformer')

    def forward(self, images):
        features = self.model(images)
        return features


class Lambda(torch.nn.Module):
    def __init__(self,paramslist,model,names,datasets):
        super().__init__()

        self.paramslist=paramslist
        self.model=model
        self.names=names

        #创建一个全1的张量，用于标识预训练模型的权重
        self.pretrain_lambdas=torch.ones(1,1)
        # 定义一个先验值。
        prior = 0.25
        #创造一个张量，用于标识预训练参数外的其他参数的权重，初始值为先验值
        values = [0.1516, 0.1349, 0.1391, 0.3271, 0.3037, 0.1786, 0.1492, 0.1737]
        rlambdas = torch.ones(1, len(paramslist) - 1) * prior  # (1 * tasks)
        rlambdas = torch.tensor([values])
        #将rlambdas包装为模型的张量
        self.lambdas_raw=torch.nn.Parameter(rlambdas)

        self.classifier=[]
        #初始化分类器列表
        for dataset_name in datasets:
            #遍历数据集名称列表##########################################
            classification_head=get_classification_head(args,dataset_name)
            #获取当前数据集的分类头
            layer_name='classifier_{}'.format(dataset_name)
            # 为当前数据集构造一个唯一的分类头名称。
            self.add_module(layer_name, classification_head.to(args.device))
            # 将分类头名称添加到分类器列表中。
            self.classifier.append(layer_name)

    def lambdas(self):
        # 定义一个函数，用于获取当前的权重。
        # 对原始权重进行裁剪，确保其值在[0.0, 1.0]范围内。
        task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
        # 将预训练权重和任务权重拼接起来。
        lambdass = torch.cat((self.pretrain_lambdas, task_lambdas), 1)
        return lambdass

    def collect_trainable_params(self):
        # 定义一个函数，用于收集可训练的参数。
        return [self.lambdas_raw]

    def get_classification_head(self, dataset_name):
        # 定义一个函数，根据数据集名称获取对应的分类头。
        # 构造分类头的名称。
        layer_name = 'classifier_{}'.format(dataset_name)
        # 通过名称获取分类头实例。
        classification_head = getattr(self, layer_name)
        return classification_head

    def get_image_encoder(self):
        # 定义一个函数，用于获取图像编码器。
        # 获取当前的权重。
        alph = self.lambdas()
        # 根据权重计算每个参数的加权和。
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        # 将参数移动到GPU上。
        params = tuple(p for p in params)#tuple(p.cuda(0) for p in params)
        # 调用load_weights函数，将计算得到的参数加载到模型中。
        load_weights(self.model, self.names, params)
        return self.model

    def forward(self, inp, dataset_name):
        # 获取当前的权重。
        alph = self.lambdas()
        # 根据权重计算每个参数的加权和。
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        # 将参数移动到GPU上。
        params = tuple(p for p in params)#tuple(p.cuda(0) for p in params)
        # 调用load_weights函数，将计算得到的参数加载到模型中。
        load_weights(self.model, self.names, params)
        # 使用配置好的模型实例对输入进行处理，获取特征。
        feature = self.model(inp)

        # 构造当前数据集对应的分类头的名称
        layer_name = 'classifier_{}'.format(dataset_name)
        # 通过名称获取分类头实例
        classification_head = getattr(self, layer_name)
        # 使用分类头对特征进行处理，获取输出。
        out = classification_head(feature)

        return out

def softmax_entropy(x):
    # 定义一个函数，计算softmax熵。
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

#def main():



if __name__=="__main__":
    models = ['ViT-B-32', 'ViT-B-16']  # , 'ViT-L-14'
    # 设定模型
    model=models[1]
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

    args = parse_arguments()
    args.device='cpu'
    args.finetuning_mode = "linear"
    args.data_location = 'E:\\Chenzebin\\data'
    args.model = model
    args.save = f'checkpoints/{model}'
    pretrained_checkpoint = f'checkpoints/{model}/DTDVal/linear_zeroshot.pt'
    args.logs_path = 'logs/' + model
    # finetuned_checkpoint = f'checkpoints/{model}/{dataset}/finetuned.pt'

    # 获取当前时间，并格式化为字符串。
    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log = create_log_dir(args.logs_path, 'log_{}_ada_LinearTA.txt'.format(str_time_))
    args.log = log

    # 对于每个数据集，创建一个任务向量
    task_vectors = [LinearizedTaskVector(pretrained_checkpoint,
                                         "checkpoints/" + model + "/" + dataset_name + "Val/linear_finetuned.pt")
                    for dataset_name in datasets]

    pretrained_model = LinearizedImageEncoder.load(pretrained_checkpoint)
    pretrained_model_dic = pretrained_model.state_dict()

    # 创建ModelWrapper实例，包装预训练模型
    model = ModelWrapper(pretrained_model, datasets)
    model = model.to(args.device)
    # 调用make_functional函数，使模型成为函数式模型，并获取参数名称列表。
    _, names = make_functional(model)

    paramslist = []
    # 将预训练模型的参数作为第一个元素添加到参数列表中。
    paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in pretrained_model_dic.items())]  # pretrain
    # 将每个任务向量的参数添加到参数列表中
    paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.vector.items()) for i, tv in
                   enumerate(task_vectors)]  # task vectors
    torch.cuda.empty_cache()

    # 创建Lambda实例，用于多任务学习
    mtl_model = Lambda(paramslist, model, names, datasets)
    print('init lambda:')
    # 打印初始化的权重。
    print(mtl_model.lambdas())
    print('collect_trainable_params:')
    # 打印可训练的参数。
    print(list(mtl_model.collect_trainable_params()))

    # 训练轮次
    epochs = 500
    # 设置优化器
    optimizer = torch.optim.Adam(mtl_model.collect_trainable_params(), lr=1e-3, betas=(0.9, 0.999),
                                 weight_decay=0.)

    # 初始化总准确率
    # Total_ACC = 0.
    # for dataset_name in datasets:
    #     image_encoder = mtl_model.get_image_encoder()
    #     classification_head = mtl_model.get_classification_head(dataset_name)
    #     metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, dataset_name, args)
    #     Total_ACC += metrics['top1']
    #     # 将评估结果记录到日志中。
    #     log.info('Eval: init: ' + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))
    #
    # # 记录平均准确率到日志中。
    # log.info('Eval: init: ' + ' Avg ACC:' + str(Total_ACC / len(datasets)) + '\n')

    for epoch in range(epochs):
        print("-------------------第{}轮------------------".format(epoch))
        losses = 0
        for dataset_name in datasets:
            print("在{}数据集上".format(dataset_name))
            dataset = get_dataset(dataset_name, pretrained_model.val_preprocess, location=args.data_location,
                                  batch_size=20)
            dataloader = get_dataloader_shuffle(dataset)

            for i, data in enumerate(tqdm.tqdm(dataloader)):
                data = maybe_dictionarize(data)
                x = data['images'].to(args.device)
                y = data['labels'].to(args.device)

                outputs = mtl_model(x, dataset_name)
                loss = softmax_entropy(outputs).mean(0)
                losses += loss

                if i > 0:  # Execute only one step
                    break

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if ((epoch + 1) % 10) == 0:
            log.info(str(list(mtl_model.lambdas().data)))

            # Total_ACC = 0.
            # for dataset_name in datasets:
            #     image_encoder = mtl_model.get_image_encoder()
            #     classification_head = mtl_model.get_classification_head(dataset_name)
            #     metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, dataset_name, args)
            #     Total_ACC += metrics['top1']
            #     log.info(
            #         'Eval: Epoch: ' + str(epoch) + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))
            # log.info('Eval: Epoch: ' + str(epoch) + ' Avg ACC:' + str(Total_ACC / len(datasets)) + '\n')