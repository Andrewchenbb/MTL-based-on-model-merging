import abc# 导入抽象基类库
import os

import torch
import torch.nn as nn
from functorch import jvp, make_functional_with_buffers# 导入用于函数化模型和计算雅可比向量积的库

from src.modeling import ImageEncoder
from src.utils import DotDict # 从项目的utils模块导入DotDict工具类


class LinearizedModel(nn.Module):
    """Creates a linearized version of a nn.Module.

    The linearized version of a model is a proper PyTorch model and can be
    trained as any other nn.Module.

    用于创建神经网络模型的线性化版本，这个线性化版本本身是一个正常的PyTorch模型，
    可以像其他nn.Module一样进行训练。

    Args:
        model (nn.Module): The model to linearize. The trainable parameters of
            the linearized model will be initialized to the parameters of this
            model.
        init_model (nn.Module): A model of the same type as `model` containing
            the parameters around which the model is initialized. If not
            provided, `model` is used as the initialization model.
    """

    def __init__(self, model: nn.Module, init_model: nn.Module = None) -> None:
        """Initializes the linearized model."""
        super().__init__()
        if init_model is None:
            init_model = model

        # 将模型转换为函数式，以便进行线性化处理
        func0, params0, self.buffers0 = make_functional_with_buffers(
            init_model.eval(), disable_autograd_tracking=True
        )
        self.func0 = lambda params, x: func0(params, self.buffers0, x)

        _, params, _ = make_functional_with_buffers(
            model, disable_autograd_tracking=True
        )

        # 设置模型参数，并将初始模型的参数设置为不可训练
        self.params = nn.ParameterList(params)
        self.params0 = nn.ParameterList(params0)
        self._model_name = model.__class__.__name__

        # The intial parameters are not trainable.
        for p in self.params0:
            p.requires_grad = False

        # The params are.
        for p in self.params:
            p.requires_grad = True

    def __call__(self, x) -> torch.Tensor:
        """Computes the linearized model output using a first-order Taylor decomposition."""
        """使用一阶泰勒分解计算线性化模型的输出。"""
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        out, dp = jvp(
            lambda param: self.func0(param, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return out + dp


class LinearizedImageEncoder(abc.ABC, nn.Module):
    """Creates a linearized version of an image encoder.
    创建图像编码器的线性化版本。
    """

    def __init__(
        self, args=None, keep_lang=False, image_encoder=None, init_encoder=None
    ):
        super().__init__()
        if image_encoder is None:
            image_encoder = ImageEncoder(args, keep_lang)
        if init_encoder is None:
            init_encoder = image_encoder

        # Copy the attributes from the image encoder.
        # 复制图像编码器的属性
        self.train_preprocess = image_encoder.train_preprocess
        self.val_preprocess = image_encoder.val_preprocess
        self.cache_dir = image_encoder.cache_dir

        self._model_name = self._get_name(args.model)
        self.model = LinearizedModel(init_model=init_encoder, model=image_encoder)

    def _get_name(self, model_name):
        """处理模型名称，以支持预训练模型的加载。"""
        if "__pretrained__" in model_name:
            model_name, _ = model_name.split("__pretrained__", "")
        return model_name

    def forward(self, x):
        # use the taylorized version of the model.
        return self.model(x)

    def __call__(self, x):
        return self.forward(x)

    def save(self, filename):
        """Saves the linearized image encoder.

        We save the model name in the state dict so that we can load the
        correct model when loading the linearized image encoder. Directly using
        torch.save would not work becuse func0 is not serializable.

        Args:
            filename (str): The path to save the taylorized image encoder.
        """
        if os.path.dirname(filename) != "":
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        state_dict = self.state_dict()
        state_dict["model_name"] = self._model_name

        torch.save(state_dict, filename)

    @classmethod
    def load(cls, filename):
        """Loads a linearized image encoder.

        It first loads the state dict with the model name and then creates the
        correct model and loads the state dict.

        Args:
            filename (str): The path to the taylorized image encoder.

        Returns:
            LinearizedImageEncoder: The loaded taylorized image encoder.
        """
        print(f"Loading image encoder from {filename}")
        state_dict = torch.load(filename, map_location="cpu")

        # ImageEncoder expects a DotDict
        args = DotDict({"model": state_dict["model_name"]})
        taylorized_encoder = cls(args)

        # Remove the model name from the state dict so that we can load the
        # model.
        state_dict.pop("model_name")
        taylorized_encoder.load_state_dict(state_dict)
        return taylorized_encoder
