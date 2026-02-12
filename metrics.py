import math, os
from pathlib import Path
from typing import Optional
from common.torch import to_numpy
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
import nibabel.quaternions as nq
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.type_conv import to_o3d_pcd, to_array


def se3_compare(a, b):
    combined = se3_cat(a, se3_inv(b))

    trace = combined[..., 0, 0] + combined[..., 1, 1] + combined[..., 2, 2]
    rre = torch.acos(torch.clamp(0.5 * (trace - 1), -1., 1.)) \
                  * 180 / math.pi
    rte = torch.norm(combined[..., :, 3], dim=-1)

    return rre, rte


def se3_init(rot=None, trans=None):

    assert rot is not None or trans is not None

    if rot is not None and trans is not None:
        pose = torch.cat([rot, trans], dim=-1)
    elif rot is None:  # rotation not provided: will set to identity
        pose = F.pad(trans, (3, 0))
        pose[..., 0, 0] = pose[..., 1, 1] = pose[..., 2, 2] = 1.0
    elif trans is None:  # translation not provided: will set to zero
        pose = F.pad(rot, (0, 1))

    return pose


def se3_cat(a, b):
    """Concatenates two SE3 transforms"""
    rot_a, trans_a = a[..., :3, :3], a[..., :3, 3:4]
    rot_b, trans_b = b[..., :3, :3], b[..., :3, 3:4]

    rot = rot_a @ rot_b
    trans = rot_a @ trans_b + trans_a
    dst = se3_init(rot, trans)
    return dst


def se3_inv(pose):
    """Inverts the SE3 transform"""
    rot, trans = pose[..., :3, :3], pose[..., :3, 3:4]
    irot = rot.transpose(-1, -2)
    itrans = -irot @ trans
    return se3_init(irot, itrans)


def apply_transform(points: np.ndarray, transform: np.ndarray, normals: Optional[np.ndarray] = None):
    # transform: (3, 4)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = np.matmul(points, rotation.T) + translation
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    else:
        return points


def compute_inlier_ratio(src, src_corr, transform, positive_radius=0.1):
    r"""Computing the inlier ratio between a set of correspondences."""
    # (N, 3); (3, 4)
    src = to_numpy(src)
    src_corr = to_numpy(src_corr)
    transform = to_numpy(transform)

    src_trans = apply_transform(src, transform)
    residuals = np.sqrt(((src_corr - src_trans) ** 2).sum(1))
    inlier_ratio = np.mean(residuals < positive_radius)
    return inlier_ratio


def registration_recall(gt_rotations, rotations, gt_translations, translations, cfg=None):
    if isinstance(gt_rotations, np.ndarray):
        gt_rotations = torch.from_numpy(gt_rotations)
        gt_translations = torch.from_numpy(gt_translations)
    if isinstance(rotations, np.ndarray):
        rotations = torch.from_numpy(rotations)
        translations = torch.from_numpy(translations)

    transformer_gt = torch.cat([gt_rotations, gt_translations.reshape(-1, 3, 1)], dim=-1)
    transformer_pred = torch.cat([rotations, translations.reshape(-1, 3, 1)], dim=-1)
    rre, rte = se3_compare(transformer_pred, transformer_gt)
    if cfg:
        recall = torch.logical_and(torch.lt(rre, cfg.reg_success_thresh_rot), torch.lt(rte, cfg.reg_success_thresh_trans)).float()
    else:
        recall = torch.logical_and(torch.lt(rre, 10), torch.lt(rte, 0.1)).float()
    # rre = torch.mean(rre)
    # rte = torch.mean(rte)
    rre = torch.median(rre)
    rte = torch.median(rte)
    recall = torch.mean(recall)

    return rre, rte, recall


def registration_recall_(gt_rotations, rotations, gt_translations, translations):
    if isinstance(gt_rotations, np.ndarray):
        gt_rotations = torch.from_numpy(gt_rotations)
        gt_translations = torch.from_numpy(gt_translations)
    if isinstance(rotations, np.ndarray):
        rotations = torch.from_numpy(rotations)
        translations = torch.from_numpy(translations)

    transformer_gt = torch.cat([gt_rotations, gt_translations.reshape(-1, 3, 1)], dim=-1)
    transformer_pred = torch.cat([rotations, translations.reshape(-1, 3, 1)], dim=-1)
    rre, rte = se3_compare(transformer_pred, transformer_gt)
    recall = torch.logical_and(torch.lt(rre, 10), torch.lt(rte, 0.1)).float()
    rre = torch.mean(rre)
    rte = torch.mean(rte)
    recall = torch.mean(recall)

    return rre, rte, recall


def calculate_R_msemae(r1, r2, seq='zyx', degrees=True):
    '''
    Calculate mse, mae euler angle error.
    :param r1: shape=(B, 3, 3), pred
    :param r2: shape=(B, 3, 3), gt
    :return:
    '''
    if isinstance(r1, torch.Tensor):
        r1 = r1.cpu().detach().numpy()
    if isinstance(r2, torch.Tensor):
        r2 = r2.cpu().detach().numpy()
    assert r1.shape == r2.shape
    eulers1, eulers2 = [], []
    for i in range(r1.shape[0]):
        euler1 = Rotation.from_matrix(r1[i]).as_euler(seq=seq, degrees=degrees)
        euler2 = Rotation.from_matrix(r2[i]).as_euler(seq=seq, degrees=degrees)
        eulers1.append(euler1)
        eulers2.append(euler2)
    eulers1 = np.stack(eulers1, axis=0)
    eulers2 = np.stack(eulers2, axis=0)
    r_mse = np.mean((eulers1 - eulers2)**2, axis=-1)
    r_mae = np.mean(np.abs(eulers1 - eulers2), axis=-1)

    return np.mean(r_mse), np.mean(r_mae)


def calculate_t_msemae(t1, t2):
    '''
    calculate translation mse and mae error.
    :param t1: shape=(B, 3)
    :param t2: shape=(B, 3)
    :return:
    '''
    t1 = t1.reshape(-1, 3)
    t2 = t2.reshape(-1, 3)
    if isinstance(t1, torch.Tensor):
        t1 = t1.cpu().detach().numpy()
    if isinstance(t2, torch.Tensor):
        t2 = t2.cpu().detach().numpy()
    assert t1.shape == t2.shape
    t_mse = np.mean((t1 - t2) ** 2, axis=1)
    t_mae = np.mean(np.abs(t1 - t2), axis=1)
    return np.mean(t_mse), np.mean(t_mae)


def evaluate_mask(mask, mask_gt):
    # mask, mask_gt: (B, N)
    accs = []
    preciss = []
    recalls = []
    f1s = []
    for m, m_gt in zip(mask, mask_gt):
        m = m.cpu()
        m_gt = m_gt.cpu()
        # mask, mask_gt: n维
        acc = accuracy_score(m_gt, m)
        precis = precision_score(m_gt, m, zero_division=0)
        recall = recall_score(m_gt, m, zero_division=0)
        f1 = f1_score(m_gt, m)

        accs.append(acc)
        preciss.append(precis)
        recalls.append(recall)
        f1s.append(f1)
    acc = np.mean(accs)
    precis = np.mean(preciss)
    recall = np.mean(recalls)
    f1 = np.mean(f1s)

    return acc, precis, recall, f1


def get_nearest_neighbor(
    q_points: np.ndarray,
    s_points: np.ndarray,
    return_index: bool = False,
):
    r"""Compute the nearest neighbor for the query points in support points."""
    s_tree = cKDTree(s_points)
    distances, indices = s_tree.query(q_points, k=1, workers=-1)
    if return_index:
        return distances, indices
    else:
        return distances


def compute_info(src_pcd, ref_pcd, transform_gt, voxel_size=0.006):
    ref_pcd = to_o3d_pcd(to_array(ref_pcd))
    src_pcd = to_o3d_pcd(to_array(src_pcd))
    ref_pcd = ref_pcd.voxel_down_sample(0.01)
    src_pcd = src_pcd.voxel_down_sample(0.01)
    ref_points = np.asarray(ref_pcd.points)
    src_points = np.asarray(src_pcd.points)

    # compute info
    src_points = apply_transform(src_points, transform_gt.cpu())
    nn_distances, nn_indices = get_nearest_neighbor(ref_points, src_points, return_index=True)
    nn_indices = nn_indices[nn_distances < voxel_size]
    if nn_indices.shape[0] > 5000:
        nn_indices = np.random.choice(nn_indices, 5000, replace=False)
    src_corr_points = src_points[nn_indices]
    if src_corr_points.shape[0] > 0:
        g = np.zeros([src_corr_points.shape[0], 3, 6])
        g[:, :3, :3] = np.eye(3)
        g[:, 0, 4] = src_corr_points[:, 2]
        g[:, 0, 5] = -src_corr_points[:, 1]
        g[:, 1, 3] = -src_corr_points[:, 2]
        g[:, 1, 5] = src_corr_points[:, 0]
        g[:, 2, 3] = src_corr_points[:, 1]
        g[:, 2, 4] = -src_corr_points[:, 0]
        gt = g.transpose([0, 2, 1])
        gtg = np.matmul(gt, g)
        cov_matrix = gtg.sum(0)
    else:
        cov_matrix = np.zeros((6, 6))

    return cov_matrix


def computeTransformationErr(trans, info):
    """
    Computer the transformation error as an approximation of the RMSE of corresponding points.
    More informaiton at http://redwood-data.org/indoor/registration.html

    Args:
    trans (numpy array): transformation matrices [4,4]
    info (numpy array): covariance matrices of the gt transformation paramaters [6,6]

    Returns:
    p (float): transformation error
    """

    t = trans[:3, 3]
    r = trans[:3, :3]
    q = nq.mat2quat(r)
    er = np.concatenate([t, q[1:]], axis=0)
    p = er.reshape(1, 6) @ info @ er.reshape(6, 1) / info[0, 0]

    return p.item()


def evaluate_log_files(cfg, tsfm_est):
    # tsfm_est: (N, 4, 4)
    from benchmark.benchmark_predator import benchmark as benchmark_predator
    from benchmark.benchmark_dgr import benchmark_dgr
    from benchmark.benchmark_predator import write_est_trajectory

    benchmark_dir = Path('/home/zy/dataset/3DMatch/benchmarks') / cfg.benchmark
    results_dir = Path(cfg.results_dir) / cfg.benchmark
    write_est_trajectory(benchmark_dir, results_dir, tsfm_est)

    if cfg.use_dgr:
        out = benchmark_dgr(results_dir, benchmark_dir, require_individual_errors=True)
    else:
        out, recall, indiv_errs, rre, rte = benchmark_predator(results_dir, benchmark_dir,
                                                     require_individual_errors=True)
        indiv_errs.to_excel(os.path.join(cfg.results_dir, 'individual_errors.xlsx'))

    print(out)
    return recall, rre, rte


