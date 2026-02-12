import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from easydict import EasyDict
import sys, os
from utils.misc import load_config
from dataloader.modelnet import get_datasets
from dataloader.threeDMatch import get_3Dmatch_datasets, get_3DLomatch_datasets, collate_pair


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

setup_seed(3407)


def compute_euclid(x, y):
    """
        pairwise_distance
        Args:
            x: Input features of source point clouds. Size [B, c, N]
            y: Input features of source point clouds. Size [B, c, M]
        Returns:
            pair_distances: Euclidean distance. Size [B, N, M]
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)

    assert len(x.shape) == len(y.shape)
    squeeze = False
    if len(x.shape) == 2:
        squeeze = True
        x = x.unsqueeze(0)
    if len(y.shape) == 2:
        y = y.unsqueeze(0)
    xx = torch.sum(torch.mul(x, x), 1, keepdim=True)  # [b,1,n]
    yy = torch.sum(torch.mul(y, y), 1, keepdim=True)  # [b,1,n]
    # print(x.shape, y.shape)
    inner = -2*torch.matmul(x.transpose(2, 1), y)  # [b,n,n]
    pair_distance = xx.transpose(2, 1) + inner + yy  # [b,n,n]
    device = x.device
    zeros_matrix = torch.zeros_like(pair_distance, device=device)
    pair_distance_square = torch.where(pair_distance > 0.0,pair_distance,zeros_matrix)
    error_mask = torch.le(pair_distance_square, 0.0)
    pair_distances = torch.sqrt(pair_distance_square + error_mask.float()*1e-16)
    pair_distances = torch.mul(pair_distances, (1.0-error_mask.float()))
    if squeeze:
        return pair_distances.squeeze(0)
    return pair_distances


def compute_overlap_rate(src, tgt, R_gt, t_gt, threshold=0.0375):
    # src, tgt: (1, 3, M)
    # R: (1, 3, 3)
    # t: (1, 3)
    if src.shape[2] == 3:
        src = src.permute(0, 2, 1)
    if tgt.shape[2] == 3:
        tgt = tgt.permute(0, 2, 1)

    num_points = min(src.shape[-1], tgt.shape[-1])
    src_trans = torch.matmul(R_gt, src) + t_gt.reshape(t_gt.shape[0], 3, 1)
    distance = compute_euclid(src_trans[:, :3, :], tgt[:, :3, :])  # (1, M, N)
    min_dist = torch.min(distance, dim=-1)[0]  # (1, M)
    olp_mask = torch.le(min_dist, threshold)
    olp_rate = torch.sum(olp_mask) / num_points

    return olp_rate


if __name__ == '__main__':

    # dataset_type = 'modelnet'
    dataset_type = '3DMatch'  # 57.1%
    # dataset_type = '3DLoMatch'  # 27.16%

    if dataset_type == '3DMatch':
        cfg_path = 'config/3dmatch_native.yaml'
        cfg = EasyDict(load_config(cfg_path))
        train_dataset = get_3Dmatch_datasets(cfg, 'train')
        test_dataset = get_3Dmatch_datasets(cfg, 'test')
    elif dataset_type == '3DLoMatch':
        cfg_path = 'config/3dlomatch_native.yaml'
        cfg = EasyDict(load_config(cfg_path))
        train_dataset = get_3DLomatch_datasets(cfg, 'train')
        test_dataset = get_3DLomatch_datasets(cfg, 'test')
    else:
        raise "Dataset not exist!"

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,  # must be 1
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_pair
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        collate_fn=collate_pair
    )

    # count = 0
    # olp_rate = 0
    # for datas in tqdm(test_loader):
    #     R_gt = datas['R']
    #     t_gt = datas['t']
    #     src = datas['src_xyz']
    #     tgt = datas['tgt_xyz']
    #     src = src[0].unsqueeze(0)  # (1, N, 3)
    #     tgt = tgt[0].unsqueeze(0)
    #     olp_rate_ = compute_overlap_rate(src, tgt, R_gt, t_gt)
    #     olp_rate += olp_rate_
    #     count += 1
    # olp_rate = olp_rate / count
    # print('Test OOR:', olp_rate)

    count = 0
    olp_rate = 0
    for datas in tqdm(train_loader):
        R_gt = datas['R']
        t_gt = datas['t']
        src = datas['src_xyz']
        tgt = datas['tgt_xyz']
        src = src[0].unsqueeze(0)  # (1, N, 3)
        tgt = tgt[0].unsqueeze(0)
        olp_rate_ = compute_overlap_rate(src, tgt, R_gt, t_gt)
        olp_rate += olp_rate_
        count += 1
    olp_rate = olp_rate / count
    print('Train OOR:', olp_rate)


