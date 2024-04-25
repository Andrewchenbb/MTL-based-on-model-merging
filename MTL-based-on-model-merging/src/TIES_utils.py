import sys
import os, copy
import torch
#import matplotlib.pyplot as plt
import numpy as np
import re
from collections import OrderedDict
import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

## Model conversion utils
# 将模型的状态字典转换成一个参数向量，便于数学操作和处理
def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]# 删除不需要的键
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))# 排序确保一致性
    # 将每个张量展平并合并成一个长向量
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )

# 将参数向量重新转换为模型的状态字典
def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict

# 将预训练模型（PTM）的参数添加到任务向量（TV）中，用于更新或调整模型的状态
def add_ptm_to_tv(tv_dict, ptm_dict):
    assert set(tv_dict.keys()) == set(
        ptm_dict.keys()
    ), "Differing parameter names in models."
    final_dict = copy.deepcopy(tv_dict)
    for k, v in ptm_dict.items():
        final_dict[k] = tv_dict[k] + v
    return final_dict

# 检查所有传入的模型的参数名是否一致
def check_parameterNamesMatch(checkpoints):
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )

# 检查两个状态字典是否完全相等
def check_state_dicts_equal(state_dict1, state_dict2):
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True



## TIES MERGING UTILS
# 筛选矩阵中的顶部 K% 的值，用于重置阈值操作
def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    # 获取 M 的原始形状，用于后续恢复形状
    original_shape = M.shape
    # 如果 M 是一维的，将其扩展为二维张量，方便统一处理
    if M.dim() == 1:
        M = M.unsqueeze(0)

    # 获取 M 的维度信息，n 是行数，d 是列数
    n, d = M.shape
    # 计算每行中应该保留的元素数目 k
    k = int(d * K)
    # 实际上我们需要保留的是顶部的 k 个元素，所以需要计算 d - k
    k = d - k  # Keep top k elements instead of bottom k elements# 保留顶部 K% 的元素

    # Find the k-th smallest element by magnitude for each row
    # 计算每一行的第 k 小的值，abs() 确保我们按元素的绝对值来计算
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    # 创建一个掩码张量，其中大于或等于第 k 小值的元素为 True
    mask = M.abs() >= kth_values
    # 如果 M 被扩展了维度，现在需要将其压缩回原始的维度形状
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    # 如果函数调用时请求返回掩码，则将原始张量与掩码相乘后的结果、掩码的平均值、以及掩码本身一起返回
    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    # 默认仅返回原始张量与掩码相乘后的结果和掩码的平均值
    return M * final_mask, final_mask.float().mean(dim=1)

# 解决零符号问题，保持向量方向一致
def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult

# 解决整体符号问题，确保所有行的符号一致
def resolve_sign(Tensor):
    sign_to_mult = torch.sign(Tensor.sum(dim=0))
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult

# 不相交合并函数，根据给定的合并函数合并行
def disjoint_merge(Tensor, merge_func, sign_to_mult):
    # 解析聚合函数的名称，假设输入形式可能是前缀加函数名，如 "dis-sum" 中的 "sum"
    merge_func = merge_func.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    # 如果提供了符号数组 sign_to_mult，则基于这些符号选择对应的张量元素进行聚合
    if sign_to_mult is not None:
        # sign_to_mult > 0 判断，对于每个位置i，如果 sign_to_mult[i] > 0，选择 Tensor[i] > 0 的元素，否则选择 Tensor[i] < 0 的元素
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )
        # 根据上述条件筛选出的 rows_to_keep，通过乘法操作仅保留应该被保留的元素，其余位置为零
        selected_entries = Tensor * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    # 如果没有提供符号数组，则选择张量中所有非零元素进行聚合
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep

    # 根据提供的聚合函数名执行对应的聚合操作
    if merge_func == "mean":
        # 计算每列中非零元素的数量
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        # 对选中的元素求和后除以非零元素的数量，计算平均值
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(non_zero_counts, min=1)
    elif merge_func == "sum":
        # 直接对选中的元素进行求和
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        # 计算选中元素的绝对值的最大值
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        # 将最大值乘以对应的符号，以恢复原始符号
        disjoint_aggs *= sign_to_mult
    else:
        # 如果提供的聚合函数名不被支持，抛出错误
        raise ValueError(f"Merge method {merge_func} is not defined.")
    # 返回聚合后的结果
    return disjoint_aggs


def ties_merging(
        flat_task_checks,# 传入的任务向量，每个向量代表一个任务的参数集
        reset_thresh=None,# 重置阈值，用于确定在聚合前保留向量中的哪些元素
        merge_func="",# 指定聚合函数，如 "sum", "mean" 或 "max"
):
    # 克隆传入的任务向量，以保持原始数据不变
    all_checks = flat_task_checks.clone()

    # 使用 topk_values_mask 函数筛选每个任务向量中重要的元素（基于 reset_thresh）
    # 这里的 topk_values_mask 函数将会返回筛选后的向量和其他可能的输出（这里未使用）
    updated_checks, *_ = topk_values_mask(
        all_checks, K=reset_thresh, return_mask=False
    )
    # 打印输出，标识正在解决符号问题
    print(f"RESOLVING SIGN")

    # 解决并统一向量中的符号问题，以确保在聚合过程中符号的一致性
    final_signs = resolve_sign(updated_checks)
    # 确保 final_signs 不为空，否则说明 resolve_sign 函数没有正确执行
    assert final_signs is not None

    # 打印输出，标识正在进行不相交聚合操作
    print(f"Disjoint AGGREGATION: {merge_func}")
    # 使用 disjoint_merge 函数对筛选并修正符号后的任务向量进行聚合
    # 这个函数将基于提供的聚合函数和符号向量来合并向量中的元素
    merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)

    return merged_tv


def disjoint_merge_split(Tensor, merge_func, sign_to_mult):
    merge_func = merge_func.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )
        selected_entries = Tensor * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep
    #仅在sum下使用
    if merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")
    #不仅执行聚合操作，还返回聚合前选择的元素
    return selected_entries, disjoint_aggs

# 结合所有功能的主函数，用于任务向量的合并
def ties_merging_split(
        flat_task_checks,
        reset_thresh=None,
        merge_func="",
):
    all_checks = flat_task_checks.clone()
    updated_checks, *_ = topk_values_mask(
        all_checks, K=reset_thresh, return_mask=False
    )
    print(f"RESOLVING SIGN")
    final_signs = resolve_sign(updated_checks)
    assert final_signs is not None

    print(f"Disjoint AGGREGATION: {merge_func}")
    selected_entries, merged_tv = disjoint_merge_split(updated_checks, merge_func, final_signs)

    return selected_entries, merged_tv
