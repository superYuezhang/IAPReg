import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import sys, os
from tqdm import tqdm
from easydict import EasyDict
from utils.misc import load_config
from dataloader.modelnet import get_datasets
from dataloader.threeDMatch import get_3Dmatch_datasets
from model import RegNet
from common.torch import to_cuda
from metrics import registration_recall, calculate_R_msemae, calculate_t_msemae, evaluate_mask,\
    compute_inlier_ratio
from compute_overlap_rate import compute_overlap_rate
from thop import profile, clever_format
import time


use_cuda = torch.cuda.is_available()
multi_gpu = False

if multi_gpu:
    gpus = [0, 1]
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
else:
    gpu_id = 0
    torch.cuda.set_device(gpu_id)

# dataset_type = 'modelnet'
dataset_type = '3DMatch'


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# 设置随机数种子
setup_seed(3407)
if dataset_type == 'modelnet':
    cfg_path = 'config/modelnet.yaml'
    cfg = EasyDict(load_config(cfg_path))
    train_dataset = get_datasets(cfg)[0]
    test_dataset = get_datasets(cfg)[1]
elif dataset_type == '3DMatch':
    cfg_path = 'config/3dmatch.yaml'
    cfg = EasyDict(load_config(cfg_path))
    train_dataset = get_3Dmatch_datasets(cfg)[0]
    test_dataset = get_3Dmatch_datasets(cfg)[1]
else:
    raise "Dataset not exist!"

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    pin_memory=False,
    drop_last=True,
)

net = RegNet(cfg).cuda()
if multi_gpu:
    net = nn.DataParallel(net.cuda(), device_ids=gpus, output_device=gpus[0])


for datas in tqdm(test_loader):
    if use_cuda:
        datas = to_cuda(datas)
    # src = datas['src_xyz']
    # tgt = datas['tgt_xyz']
    R_gt = datas['R']
    t_gt = datas['t']

    flops, params = profile(net, inputs=(datas,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops)
    print(params)

    c0 = time.perf_counter()
    p0 = time.process_time()
    outputs = net(datas)
    c1 = time.perf_counter()
    p1 = time.process_time()
    spend2 = c1 - c0
    spend3 = p1 - p0
    spend = (spend2 + spend3) / 2
    print("Inference Time：{}ms".format(spend * 1000))
    break

