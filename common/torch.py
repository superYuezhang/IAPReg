import logging
import os
import pdb
import shutil
import sys
import time
import traceback
import numpy as np
import torch


def dict_all_to_device(tensor_dict, device):
    """Sends everything into a certain device """
    for k in tensor_dict:
        if isinstance(tensor_dict[k], torch.Tensor):
            tensor_dict[k] = tensor_dict[k].to(device)

try:
    import torch_geometric
    _torch_geometric_exists = True
except ImportError:
    _torch_geometric_exists = False

def all_to_device(data, device):
    """Sends everything into a certain device """
    if isinstance(data, dict):
        for k in data:
            data[k] = all_to_device(data[k], device)
        return data
    elif isinstance(data, list):
        data = [all_to_device(d, device) for d in data]
        return data
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif _torch_geometric_exists and isinstance(data, torch_geometric.data.batch.Batch):
        return data.to(device)
    else:
        return data  # Cannot be converted


def to_cuda(x):
    r"""Move all tensors to cuda."""
    if isinstance(x, list):
        x = [to_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (to_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: to_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        x = x.cuda()
    return x


def to_numpy(tensor):
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise NotImplementedError
