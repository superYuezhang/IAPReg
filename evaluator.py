import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import sys, os

sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from easydict import EasyDict
from utils.misc import load_config
from dataloader.modelnet import get_datasets
# from dataloader.ThreeDMatchSample import get_3Dmatch_datasets
from dataloader.threeDMatch import get_3Dmatch_datasets, get_3DLomatch_datasets
from model import RegNet
from common.torch import to_cuda
from metrics import registration_recall, calculate_R_msemae, calculate_t_msemae, evaluate_mask, \
    compute_inlier_ratio, compute_info, computeTransformationErr
from compute_overlap_rate import compute_overlap_rate


use_cuda = torch.cuda.is_available()
multi_gpu = False

if multi_gpu:
    gpus = [0, 1]
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
else:
    gpu_id = 1
    torch.cuda.set_device(gpu_id)

dataset_type = 'modelnet'
# dataset_type = '3DMatch'

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
elif dataset_type == '3DLoMatch':
    cfg_path = 'config/3dmatch.yaml'
    cfg = EasyDict(load_config(cfg_path))
    train_dataset = get_3DLomatch_datasets(cfg)[0]
    test_dataset = get_3DLomatch_datasets(cfg)[1]
else:
    raise "Dataset not exist!"


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# 设置随机数种子
setup_seed(3407)

def test_one_epoch(net, test_loader):
    net.eval()
    total_loss = 0
    count = 0
    olp_rate = 0
    count_olp = 0
    inlier_ratio = 0
    fmr = 0

    with torch.no_grad():
        Rs_gt = []
        ts_gt = []
        Rs_pred = []
        ts_pred = []
        f1s_src = []
        f1s_tgt = []
        recalls_src = []
        recalls_tgt = []

        for datas in tqdm(test_loader):
            if use_cuda:
                datas = to_cuda(datas)
            # src = datas['src_xyz']
            # tgt = datas['tgt_xyz']
            R_gt = datas['R']
            t_gt = datas['t']

            outputs, loss = net(datas)
            R_pred = outputs['R']
            t_pred = outputs['t']

            src = datas['src_xyz']
            tgt = datas['tgt_xyz']
            max_points = datas['max_points'][0]
            olp_rate_ = compute_overlap_rate(max_points, src, tgt, R_gt, t_gt)
            olp_rate += olp_rate_
            count_olp += 1

            batch_size = R_gt.shape[0]
            count += batch_size
            total_loss += loss.item()

            # # 评估
            # src_mask = torch.cat(outputs['src_mask_list'], dim=0).squeeze(-1).unsqueeze(0)
            # tgt_mask = torch.cat(outputs['tgt_mask_list'], dim=0).squeeze(-1).unsqueeze(0)
            # src_mask_gt = torch.cat(outputs['src_mask_gt_list'], dim=0).squeeze(-1).unsqueeze(0)
            # tgt_mask_gt = torch.cat(outputs['tgt_mask_gt_list'], dim=0).squeeze(-1).unsqueeze(0)
            # acc_src, precis_src, recall_src, f1_src = evaluate_mask(src_mask, src_mask_gt)  # input: (B, N)
            # f1s_src.append(f1_src)
            # recalls_src.append(recall_src)
            #
            # acc_tgt, precis_tgt, recall_tgt, f1_tgt = evaluate_mask(tgt_mask, tgt_mask_gt)
            # f1s_tgt.append(f1_tgt)
            # recalls_tgt.append(recall_tgt)

            Rs_pred.append(R_pred.detach().cpu().numpy())
            ts_pred.append(t_pred.detach().cpu().numpy())
            Rs_gt.append(R_gt.detach().cpu().numpy())
            ts_gt.append(t_gt.detach().cpu().numpy())
            for i in range(batch_size):
                inlier_ratio_t = compute_inlier_ratio(outputs['src_samp_list'][i], outputs['src_corr_list'][i],
                                                      datas['transformer'][i])
                inlier_ratio += inlier_ratio_t
                fmr += float(inlier_ratio_t > 0.6)

        Rs_gt = np.concatenate(Rs_gt, axis=0)
        ts_gt = np.concatenate(ts_gt, axis=0)
        Rs_pred = np.concatenate(Rs_pred, axis=0)
        ts_pred = np.concatenate(ts_pred, axis=0)

        # recall_src = np.mean(recalls_src)
        # recall_tgt = np.mean(recalls_tgt)
        # recall = (recall_src + recall_tgt) / 2
        # f1_src = np.mean(f1s_src)
        # f1_tgt = np.mean(f1s_tgt)
        # f1 = (f1_src + f1_tgt) / 2
        recall = 0
        f1 = 0
        olp_rate = olp_rate / count_olp
        inlier_ratio = inlier_ratio / count
        fmr = fmr / count

    return olp_rate, total_loss / count, Rs_gt, ts_gt, Rs_pred, ts_pred, f1, recall, inlier_ratio, fmr


if __name__ == '__main__':

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
    )

    net = RegNet(cfg).cuda()
    if multi_gpu:
        net = nn.DataParallel(net.cuda(), device_ids=gpus, output_device=gpus[0])

    # path_checkpoint = "./checkpoint/modelnet.pth"  # 断点路径
    path_checkpoint = "./checkpoint/modelnet.pth"  # 断点路径
    # checkpoint = torch.load(path_checkpoint, map_location=lambda storage, loc: storage.cuda(gpu_id))  # 加载断点
    checkpoint = torch.load(path_checkpoint)  # 加载断点
    net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

    test_olp_rate, test_total_loss, test_Rs_gt, test_ts_gt, test_Rs_pred, test_ts_pred, test_f1, \
    test_olp_recall, test_inlier_ratio, test_fmr = test_one_epoch(net, test_loader)

    test_R_mse, test_R_mae = calculate_R_msemae(test_Rs_gt, test_Rs_pred)
    test_R_rmse = np.sqrt(test_R_mse)
    test_t_mse, test_t_mae = calculate_t_msemae(test_ts_gt, test_ts_pred)
    test_t_rmse = np.sqrt(test_t_mse)
    test_RRE, test_RtE, test_recall = registration_recall(test_Rs_gt, test_Rs_pred, test_ts_gt, test_ts_pred, cfg)

    print(
        'Test: Loss: %f, OLPRate: %f, F1: %f, OLPRecall: %f, Recall: %f, IR: %f, FMR: %f,  RRE(R): %f, RMSE(R): %f, MAE(R): %f, RRE(t): %f, RMSE(t): %f, MAE(t): %f'
        % (test_total_loss, test_olp_rate, test_f1, test_olp_recall, test_recall, test_inlier_ratio, test_fmr,
           test_RRE, test_R_rmse, test_R_mae, test_RtE, test_t_rmse, test_t_mae))




