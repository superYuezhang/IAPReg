import numpy as np
import torch
import torch.nn as nn


def mask_point_list(mask_idx, points):
    # masks: [(N1, 1), ...] : list, only include 0 and 1
    # points: [(N1, 3), ...] : list
    # return: [(N', 3), ...] : list
    new_points = []
    ele_lens = []
    B = len(points)

    for i in range(B):
        new_pc = points[i] * mask_idx[i]
        # 删除被屏蔽的0点
        temp = new_pc[:, ...] == 0
        temp = temp.cpu()
        idx = np.argwhere(temp.all(axis=1))
        new_point = np.delete(new_pc.cpu().detach().numpy(), idx, axis=0)

        new_points.append(torch.from_numpy(new_point).to(points[0].device))
        ele_lens.append(new_point.shape[0])
    return new_points, ele_lens


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


def compute_overlap_mask(src, tgt, R_gt, t_gt):
    # src, tgt: (B, 3, M)
    # R: (B, 3, 3)
    # t: (B, 3)
    if src.shape[2] == 3:
        src = src.permute(0, 2, 1)
    if tgt.shape[2] == 3:
        tgt = tgt.permute(0, 2, 1)
    t_gt = t_gt.reshape(t_gt.shape[0], 3, 1)
    overlap_radius = 0.08
    src_trans = torch.matmul(R_gt, src) + t_gt
    distance = compute_euclid(src_trans, tgt)  # (B, M, N)
    src_mask = torch.min(distance, dim=2)[0] < overlap_radius
    tgt_mask = torch.min(distance.permute(0, 2, 1), dim=2)[0] < overlap_radius
    return src_mask, tgt_mask


def compute_overlap_mask_list(src_list, tgt_list, R_gt, t_gt, overlap_radius=0.08):
    # src, tgt: List of (3, M)
    # R: (B, 3, 3)
    # t: (B, 3)
    # return: List of (M, )
    if len(t_gt.shape) == 3:
        t_gt = t_gt.squeeze(-1)

    src_mask = []
    tgt_mask = []
    src_score = []
    tgt_score = []
    for i in range(len(src_list)):
        src_i = src_list[i]
        tgt_i = tgt_list[i]
        R_gt_i = R_gt[i]
        t_gt_i = t_gt[i]
        if src_i.shape[1] == 3:
            src_i = src_i.permute(1, 0)
        if tgt_i.shape[1] == 3:
            tgt_i = tgt_i.permute(1, 0)

        src_trans = torch.matmul(R_gt_i, src_i) + t_gt_i.reshape(3, 1)
        distance = compute_euclid(src_trans, tgt_i)  # (M, N)

        src_dist = torch.min(distance, dim=1)[0]
        src_mask_i = src_dist < overlap_radius
        # src_dist = torch.where(src_dist < 0.03, 0, src_dist)
        src_score_i = 1 - src_dist
        # src_score_i = torch.where(src_score_i < 0, 0, src_score_i)

        tgt_dist = torch.min(distance.permute(1, 0), dim=1)[0]
        tgt_mask_i = tgt_dist < overlap_radius
        # tgt_dist = torch.where(tgt_dist < 0.03, 0, tgt_dist)
        tgt_score_i = 1 - tgt_dist
        # tgt_score_i = torch.where(tgt_score_i < 0, 0, tgt_score_i)

        src_score.append(src_score_i)
        tgt_score.append(tgt_score_i)
        src_mask.append(src_mask_i)
        tgt_mask.append(tgt_mask_i)

    return src_mask, tgt_mask, src_score, tgt_score


def compute_corr(src, tgt, R_gt, t_gt):
    # src, tgt: (B, 3, M)
    # R: (B, 3, 3)
    # t: (B, 3)
    if src.shape[2] == 3:
        src = src.permute(0, 2, 1)
    if tgt.shape[2] == 3:
        tgt = tgt.permute(0, 2, 1)
    if len(t_gt.shape) == 3:
        t_gt = t_gt.squeeze(-1)
    src_trans = torch.matmul(R_gt, src) + t_gt.reshape(t_gt.shape[0], 3, -1)
    distance = compute_euclid(src_trans[:, :3, :], tgt[:, :3, :])  # (B, M, N)
    tgt_corr_idx = torch.max(distance, dim=2)[1]

    return tgt_corr_idx


def compute_corr_list(src, tgt, R_gt, t_gt):
    # src, tgt: [(3, M), ]
    # R: [(3, 3), ]
    # t: [(, 3), ]
    tgt_idx_list = []
    dists = []
    for i in range(len(src)):
        src_i = src[i]
        tgt_i = tgt[i]
        if src_i.shape[0] not in [3, 6]:
            src_i = src_i.permute(1, 0)
            tgt_i = tgt_i.permute(1, 0)
        src_trans = torch.matmul(R_gt[i], src_i[:3, :]) + t_gt[i].reshape(3, -1)
        distance = compute_euclid(src_trans[:3, :], tgt_i[:3, :])  # (M, N)
        dist, idx = torch.min(distance, dim=1)
        tgt_idx_list.append(idx)
        dists.append(dist)

    return tgt_idx_list, dists


def batch_transform(batch_pc, batch_R, batch_t=None):
    '''
    :param batch_pc: shape=(B, 3, N)
    :param batch_R: shape=(B, 3, 3)
    :param batch_t: shape=(B, 3)
    :return: shape(B, N, 3)
    '''
    batch_size = batch_pc.shape[0]
    transformed_pc = torch.matmul(batch_R, batch_pc)
    if batch_t is not None:
        transformed_pc = transformed_pc + batch_t.reshape(batch_size, 3, 1)
    return transformed_pc


def batch_quat2mat(batch_quat):
    '''
    :param batch_quat: shape=(B, 4)
    :return:
    '''
    batch_quat = batch_quat.squeeze()
    w, x, y, z = batch_quat[:, 0], batch_quat[:, 1], batch_quat[:, 2], \
                 batch_quat[:, 3]
    device = batch_quat.device
    B = batch_quat.size()[0]
    R = torch.zeros(dtype=torch.float, size=(B, 3, 3)).to(device)
    R[:, 0, 0] = 1 - 2 * y * y - 2 * z * z
    R[:, 0, 1] = 2 * x * y - 2 * z * w
    R[:, 0, 2] = 2 * x * z + 2 * y * w
    R[:, 1, 0] = 2 * x * y + 2 * z * w
    R[:, 1, 1] = 1 - 2 * x * x - 2 * z * z
    R[:, 1, 2] = 2 * y * z - 2 * x * w
    R[:, 2, 0] = 2 * x * z - 2 * y * w
    R[:, 2, 1] = 2 * y * z + 2 * x * w
    R[:, 2, 2] = 1 - 2 * x * x - 2 * y * y
    return R


def gather_points(points, inds):
    '''
    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]


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


def pad_list(sequences, require_padding_mask=True, require_lens=True):
    """List of sequences to padded sequences
    Args:
        sequences: List of sequences [(N, D)]
        require_padding_mask:
    Returns:
           padded sequence has shape (B, N_max, D)
    """
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    padding_mask = None
    padding_lens = None

    if require_padding_mask:
        B = len(sequences)
        seq_lens = list(map(len, sequences))  # get len(num_points) of each element
        padding_mask = torch.zeros((padded.shape[0], padded.shape[1]), dtype=torch.bool, device=padded.device)  # (batch, N)
        for i, l in enumerate(seq_lens):
            padding_mask[i, l:] = True  # True is padding element

    if require_lens:
        padding_lens = [seq.shape[0] for seq in sequences]

    return padded, padding_mask, padding_lens


def unpad_list(padded, seq_lens):
    """Reverse of pad_sequence, 填充的部分去掉"""
    # padded: (B, N, C)
    # if len(padded.shape) == 2:
    #     padded = padded.unsqueeze(-1)
    sequences = [padded[b, :seq_lens[b], ...] for b in range(len(seq_lens))]
    return sequences


def split_src_tgt(feats, stack_lengths, dim=0):
    if isinstance(stack_lengths, torch.Tensor):
        stack_lengths = stack_lengths.tolist()

    B = len(stack_lengths) // 2
    separate = torch.split(feats, stack_lengths, dim=dim)
    return separate[:B], separate[B:]


def split_to_list(feats, stack_lengths, dim=0):
    # feats: (N, C)
    # length: [N1, N2, ...]
    if isinstance(stack_lengths, torch.Tensor):
        stack_lengths = stack_lengths.tolist()

    separate = list(torch.split(feats, stack_lengths, dim=dim))

    return separate


def normal_data(input_tensor, dim=0, normal_type='maxmin'):
    if normal_type == 'maxmin':
        # max-min normal
        input_min = torch.min(input_tensor, dim=dim, keepdim=True).values
        input_max = torch.max(input_tensor, dim=dim, keepdim=True).values
        result = (input_tensor - input_min) / (input_max - input_min + 1e-5)
    elif normal_type == 'meanstd':
        # mean-std normal
        mean = torch.mean(input_tensor, dim=dim, keepdim=True)
        std = torch.std(input_tensor, dim=dim, keepdim=True)
        result = (input_tensor - mean) / (std + 1e-5)
    else:
        raise 'Normal must be in [maxmin, meanstd]'
    return result


if __name__ == '__main__':
    a = torch.tensor([1,1,1, 1,1,1, 1,1,1, 10, 10])
    b = normal_data(a, dim=0)
    print(b)


