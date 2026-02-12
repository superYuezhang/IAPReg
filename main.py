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
from dataloader.threeDMatch import get_3Dmatch_datasets
from model import RegNet
from common.torch import to_cuda
from metrics import registration_recall, calculate_R_msemae, calculate_t_msemae, evaluate_mask,\
    compute_inlier_ratio


use_cuda = torch.cuda.is_available()

gpu_id = 1
torch.cuda.set_device(gpu_id)

dataset_type = 'modelnet'


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

            # src = datas['src_xyz']
            # tgt = datas['tgt_xyz']
            # max_points = datas['max_points'][0]
            # olp_rate_ = compute_overlap_rate(max_points, src, tgt, R_gt, t_gt)
            # olp_rate += olp_rate_
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


def train_one_epoch(net, opt, train_loader):
    net.train()

    Rs_gt = []
    ts_gt = []
    Rs_pred = []
    ts_pred = []

    inlier_ratio = 0
    fmr = 0
    total_loss = 0
    count = 0
    olp_rate = 0
    count_olp = 0
    for datas in tqdm(train_loader):
        if use_cuda:
            datas = to_cuda(datas)
        # src = datas['src_xyz']
        # tgt = datas['tgt_xyz']

        R_gt = datas['R']
        t_gt = datas['t']
        outputs, loss = net(datas)
        R_pred = outputs['R']
        t_pred = outputs['t']

        # src = datas['src_xyz']
        # tgt = datas['tgt_xyz']
        # max_points = datas['max_points'][0]
        # olp_rate_ = compute_overlap_rate(max_points, src, tgt, R_gt, t_gt)
        # olp_rate += olp_rate_
        count_olp += 1

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)
        opt.step()

        batch_size = R_gt.shape[0]
        count += batch_size
        total_loss += loss.item()

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

    recall = 0
    f1 = 0
    olp_rate = olp_rate / count_olp
    inlier_ratio = inlier_ratio / count
    fmr = fmr / count

    return olp_rate, total_loss / count, Rs_gt, ts_gt, Rs_pred, ts_pred, f1, recall, inlier_ratio, fmr


if __name__ == '__main__':

    best_loss = np.inf
    best_R_mse = np.inf
    best_R_rmse = np.inf
    best_R_mae = np.inf
    best_t_mse = np.inf
    best_t_rmse = np.inf
    best_t_mae = np.inf
    best_RRE = np.inf
    best_RtE = np.inf
    best_recall = -1
    best_f1 = 0
    best_olp_recall = 0
    best_olp_rate = 1
    best_inlier_rate = 0
    best_fmr = 0

    if dataset_type == 'modelnet':
        cfg_path = 'config/modelnet.yaml'
        cfg = EasyDict(load_config(cfg_path))
        train_dataset = get_datasets(cfg)[0]
        test_dataset = get_datasets(cfg)[1]
    else:
        raise "Dataset not exist!"

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    net = RegNet(cfg).cuda()

    opt = optim.AdamW(params=net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # 动态调整学习率
    # scheduler = MultiStepLR(opt, milestones=[100, 200, 300], gamma=0.1)
    scheduler = MultiStepLR(opt, milestones=[70, 140], gamma=cfg.scheduler_param)
    start_epoch = -1
    RESUME = False  # 是否加载模型继续上次训练

    if RESUME:
        path_checkpoint = "./checkpoint/ckpt%s.pth" % (str(dataset_type))  # 断点路径
        # checkpoint = torch.load(path_checkpoint, map_location=lambda storage, loc: storage.cuda(gpu_id))  # 加载断点
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        # scheduler.load_state_dict(checkpoint["lr_step"])
        opt.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        # 加载上次best结果
        best_loss = checkpoint['best_loss']

        best_R_rmse = checkpoint['best_RMSE(R)']
        best_R_mae = checkpoint['best_MAE(R)']
        best_t_rmse = checkpoint['best_RMSE(t)']
        best_t_mae = checkpoint['best_MAE(t)']
        best_f1 = checkpoint['best_f1']
        best_olp_recall = checkpoint['best_olp_recall']
        best_RRE = checkpoint['best_RRE']
        best_RtE = checkpoint['best_RtE']
        best_inlier_rate = checkpoint['best_inlier_rate']
        best_fmr = checkpoint['best_fmr']
        best_olp_rate = checkpoint['best_olp_rate']
        best_recall = checkpoint['best_recall']
        scheduler.load_state_dict(checkpoint['scheduler'])
        scheduler.last_epoch = start_epoch

    for epoch in range(start_epoch + 1, cfg.epochs):

        train_olp_rate, train_total_loss, train_Rs_gt, train_ts_gt, train_Rs_pred, train_ts_pred, train_f1, \
        train_olp_recall, train_inlier_ratio, train_fmr = train_one_epoch(net, opt, train_loader)

        test_olp_rate, test_total_loss, test_Rs_gt, test_ts_gt, test_Rs_pred, test_ts_pred, test_f1, \
        test_olp_recall, test_inlier_ratio, test_fmr = test_one_epoch(net, test_loader)

        train_R_mse, train_R_mae = calculate_R_msemae(train_Rs_gt, train_Rs_pred)
        train_R_rmse = np.sqrt(train_R_mse)
        train_t_mse, train_t_mae = calculate_t_msemae(train_ts_gt, train_ts_pred)
        train_t_rmse = np.sqrt(train_t_mse)
        train_RRE, train_RtE, train_recall = registration_recall(train_Rs_gt, train_Rs_pred, train_ts_gt, train_ts_pred,
                                                                 cfg)

        test_R_mse, test_R_mae = calculate_R_msemae(test_Rs_gt, test_Rs_pred)
        test_R_rmse = np.sqrt(test_R_mse)
        test_t_mse, test_t_mae = calculate_t_msemae(test_ts_gt, test_ts_pred)
        test_t_rmse = np.sqrt(test_t_mse)
        test_RRE, test_RtE, test_recall = registration_recall(test_Rs_gt, test_Rs_pred, test_ts_gt, test_ts_pred, cfg)

        # print('RRE and RMSE:', train_RRE, train_RtE, train_R_rmse, train_t_rmse)
        # print('RRE and RMSE:', test_RRE, test_RtE, test_R_rmse, test_t_rmse)
        # scheduler.step(epoch)
        scheduler.step()

        if test_RRE < best_RRE:
            best_loss = test_total_loss

            best_R_rmse = test_R_rmse
            best_R_mae = test_R_mae

            best_t_rmse = test_t_rmse
            best_t_mae = test_t_mae
            best_RRE = test_RRE
            best_RtE = test_RtE
            best_recall = test_recall
            best_f1 = test_f1
            best_olp_recall = test_olp_recall
            best_olp_rate = test_olp_rate
            best_inlier_rate = test_inlier_ratio
            best_fmr = test_fmr
            # 保存最好的checkpoint
            checkpoint_best = {
                "net": net.state_dict(),
            }
            if not os.path.isdir("./checkpoint"):
                os.mkdir("./checkpoint")
            torch.save(checkpoint_best, './checkpoint/ckpt_best%s.pth' % (str(dataset_type)))

        print('---------Epoch: %d---------' % (epoch + 1))
        print(
            'Train: Loss: %f, OLPRate: %.1f, F1: %.1f, OLPRecall: %.1f, Recall: %f, IR: %f, FMR: %f, RRE(R): %f, RMSE(R): %f, MAE(R): %f, RRE(t): %f, RMSE(t): %f, MAE(t): %f'
            % (
            train_total_loss, train_olp_rate, train_f1, train_olp_recall, train_recall, train_inlier_ratio, train_fmr,
            train_RRE, train_R_rmse, train_R_mae, train_RtE, train_t_rmse, train_t_mae))

        print(
            'Test: Loss: %f, OLPRate: %.1f, F1: %.1f, OLPRecall: %.1f, Recall: %f, IR: %f, FMR: %f,  RRE(R): %f, RMSE(R): %f, MAE(R): %f, RRE(t): %f, RMSE(t): %f, MAE(t): %f'
            % (test_total_loss, test_olp_rate, test_f1, test_olp_recall, test_recall, test_inlier_ratio, test_fmr,
               test_RRE, test_R_rmse, test_R_mae, test_RtE, test_t_rmse, test_t_mae))

        print(
            'Best: Loss: %f, OLPRate: %.1f, F1: %.1f, OLPRecall: %.1f, Recall: %f, IR: %f, FMR: %f, RRE(R): %f, RMSE(R): %f, MAE(R): %f, RRE(t): %f, RMSE(t): %f, MAE(t): %f'
            % (
            best_loss, best_olp_rate, best_f1, best_olp_recall, best_recall, best_inlier_rate, best_fmr,
            best_RRE, best_R_rmse, best_R_mae, best_RtE, best_t_rmse, best_t_mae))

        # 保存checkpoint
        checkpoint = {
            "net": net.state_dict(),
            'optimizer': opt.state_dict(),
            "epoch": epoch,
            'scheduler': scheduler.state_dict(),

            "best_loss": best_loss,
            'best_recall': best_recall,
            'best_f1': best_f1,
            'best_olp_recall': best_olp_recall,
            'best_RMSE(R)': best_R_rmse,
            'best_MAE(R)': best_R_mae,
            'best_RMSE(t)': best_t_rmse,
            'best_MAE(t)': best_t_mae,
            'best_RRE': best_RRE,
            'best_RtE': best_RtE,
            'best_olp_rate': best_olp_rate,
            'best_inlier_rate': best_inlier_rate,
            'best_fmr': best_fmr,
        }
        if not os.path.isdir("./checkpoint"):
            os.mkdir("./checkpoint")
        torch.save(checkpoint, './checkpoint/ckpt%s.pth' % (str(dataset_type)))




