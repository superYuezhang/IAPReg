import numpy as np
import torch
import torch.nn as nn
import open3d as o3d


def to_tensor(array):
    """
    Convert array to tensor
    """
    if(not isinstance(array,torch.Tensor)):
        return torch.from_numpy(array).float()
    else:
        return array


def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        if(tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor


def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd

def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.pipelines.registration.Feature()
    feats.data = to_array(embedding).T
    return feats


def mutual_selection(score_mat):
    """
    Return a {0,1} matrix, the element is 1 if and only if it's maximum along both row and column

    Args: np.array()
        score_mat:  [B,N,N]
    Return:
        mutuals:    [B,N,N]
    """
    score_mat = to_array(score_mat)
    if (score_mat.ndim == 2):
        score_mat = score_mat[None, :, :]

    mutuals = np.zeros_like(score_mat)
    for i in range(score_mat.shape[0]):  # loop through the batch
        c_mat = score_mat[i]
        flag_row = np.zeros_like(c_mat)
        flag_column = np.zeros_like(c_mat)

        max_along_row = np.argmax(c_mat, 1)[:, None]
        max_along_column = np.argmax(c_mat, 0)[None, :]
        np.put_along_axis(flag_row, max_along_row, 1, 1)
        np.put_along_axis(flag_column, max_along_column, 1, 0)
        mutuals[i] = (flag_row.astype(np.bool)) & (flag_column.astype(np.bool))
    return mutuals.astype(np.bool)


def ransac_reg(src_pcd, tgt_pcd, src_feat, tgt_feat, mutual=False, distance_threshold=0.05, ransac_n=3):
    """
    RANSAC pose estimation with two checkers
    We follow D3Feat to set ransac_n = 3 for 3DMatch and ransac_n = 4 for KITTI.
    For 3DMatch dataset, we observe significant improvement after changing ransac_n from 4 to 3.
    """
    if (mutual):
        if (torch.cuda.device_count() >= 1):
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        src_feat, tgt_feat = to_tensor(src_feat), to_tensor(tgt_feat)
        scores = torch.matmul(src_feat.to(device), tgt_feat.transpose(0, 1).to(device)).cpu()
        selection = mutual_selection(scores[None, :, :])[0]
        row_sel, col_sel = np.where(selection)
        corrs = o3d.utility.Vector2iVector(np.array([row_sel, col_sel]).T)
        src_pcd = to_o3d_pcd(src_pcd)
        tgt_pcd = to_o3d_pcd(tgt_pcd)
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            source=src_pcd, target=tgt_pcd, corres=corrs,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))
    else:
        src_pcd = to_o3d_pcd(src_pcd)
        tgt_pcd = to_o3d_pcd(tgt_pcd)
        src_feats = to_o3d_feats(src_feat)
        tgt_feats = to_o3d_feats(tgt_feat)

        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_pcd, tgt_pcd, src_feats, tgt_feats, False, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))

    return result_ransac.transformation


class LearnableLogOptimalTransport(nn.Module):
    def __init__(self, num_iterations=100, inf=1e12):
        r"""Sinkhorn Optimal transport with dustbin parameter (SuperGlue style)."""
        super(LearnableLogOptimalTransport, self).__init__()
        self.num_iterations = num_iterations
        self.register_parameter('alpha', torch.nn.Parameter(torch.tensor(1.0)))
        self.inf = inf

    def log_sinkhorn_normalization(self, scores, log_mu, log_nu):
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(self.num_iterations):
            u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
        return scores + u.unsqueeze(2) + v.unsqueeze(1)

    def forward(self, scores, row_masks=None, col_masks=None):
        r"""Sinkhorn Optimal Transport (SuperGlue style) forward.

        Args:
            scores: torch.Tensor (B, M, N)
            row_masks: torch.Tensor (B, M)
            col_masks: torch.Tensor (B, N)

        Returns:
            matching_scores: torch.Tensor (B, M+1, N+1)
        """
        batch_size, num_row, num_col = scores.shape

        if row_masks is None:
            row_masks = torch.ones(size=(batch_size, num_row), dtype=torch.bool).cuda()
        if col_masks is None:
            col_masks = torch.ones(size=(batch_size, num_col), dtype=torch.bool).cuda()

        padded_row_masks = torch.zeros(size=(batch_size, num_row + 1), dtype=torch.bool).cuda()
        padded_row_masks[:, :num_row] = ~row_masks
        padded_col_masks = torch.zeros(size=(batch_size, num_col + 1), dtype=torch.bool).cuda()
        padded_col_masks[:, :num_col] = ~col_masks
        padded_score_masks = torch.logical_or(padded_row_masks.unsqueeze(2), padded_col_masks.unsqueeze(1))

        padded_col = self.alpha.expand(batch_size, num_row, 1)
        padded_row = self.alpha.expand(batch_size, 1, num_col + 1)
        padded_scores = torch.cat([torch.cat([scores, padded_col], dim=-1), padded_row], dim=1)
        padded_scores.masked_fill_(padded_score_masks, -self.inf)

        num_valid_row = row_masks.float().sum(1)
        num_valid_col = col_masks.float().sum(1)
        norm = -torch.log(num_valid_row + num_valid_col)  # (B,)

        log_mu = torch.empty(size=(batch_size, num_row + 1)).cuda()
        log_mu[:, :num_row] = norm.unsqueeze(1)
        log_mu[:, num_row] = torch.log(num_valid_col) + norm
        log_mu[padded_row_masks] = -self.inf

        log_nu = torch.empty(size=(batch_size, num_col + 1)).cuda()
        log_nu[:, :num_col] = norm.unsqueeze(1)
        log_nu[:, num_col] = torch.log(num_valid_row) + norm
        log_nu[padded_col_masks] = -self.inf

        outputs = self.log_sinkhorn_normalization(padded_scores, log_mu, log_nu)
        outputs = outputs - norm.unsqueeze(1).unsqueeze(2)

        return outputs

    def __repr__(self):
        format_string = self.__class__.__name__ + '(num_iterations={})'.format(self.num_iterations)
        return format_string


class WeightedSVD(nn.Module):
    def __init__(self):
        super(WeightedSVD, self).__init__()

        self.my_iter = torch.ones(1)
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def SVD(self, src, src_corr):
        # (batch, 3, n)
        squeeze = False
        if len(src.shape) == 2:
            src = src.unsqueeze(0)
            src_corr = src_corr.unsqueeze(0)
            squeeze = True
        if src.shape[2] == 3:
            src = src.permute(0, 2, 1)
        if src_corr.shape[2] == 3:
            src_corr = src_corr.permute(0, 2, 1)
        batch_size = src.shape[0]
        src_centered = src - src.mean(dim=2, keepdim=True)
        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous()).cpu()
        R = []
        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0)).contiguous()
            r_det = torch.det(r).item()
            diag = torch.from_numpy(np.array([[1.0, 0, 0],
                                              [0, 1.0, 0],
                                              [0, 0, r_det]]).astype('float32')).to(v.device)
            r = torch.matmul(torch.matmul(v, diag), u.transpose(1, 0)).contiguous()
            R.append(r)

        R = torch.stack(R, dim=0).cuda()
        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        if self.training:
            self.my_iter += 1
        if squeeze:
            return R.squeeze(0), t.squeeze(0)
        return R, t.view(batch_size, 3)

    def forward(self, src, src_corr, weight=None):
        """
                Compute rigid transforms between two point sets
                Args:
                    src: Source point clouds. Size (B, 3, N)
                    src_corr: Pseudo tgtget point clouds. Size (B, 3, N)
                    weights: Inlier confidence. (B, 1, N)
                Returns:
                    R: Rotation. Size (B, 3, 3)
                    t: translation. Size (B, 3, 1)
            """
        if weight is None:
            return self.SVD(src, src_corr)

        squeeze = False
        if len(src.shape) == 2:
            src = src.unsqueeze(0)
            src_corr = src_corr.unsqueeze(0)
            weight = weight.unsqueeze(0)
            squeeze = True
        if src.shape[2] == 3:
            src = src.permute(0, 2, 1)
        if src_corr.shape[2] == 3:
            src_corr = src_corr.permute(0, 2, 1)
        if len(weight.shape) == 2:
            weight = weight.unsqueeze(1)
        src2 = (src * weight).sum(dim=2, keepdim=True) / weight.sum(dim=2, keepdim=True)
        src_corr2 = (src_corr * weight).sum(dim=2, keepdim=True) / weight.sum(dim=2, keepdim=True)
        src_centered = src - src2
        src_corr_centered = src_corr - src_corr2
        H = torch.matmul(src_centered * weight, src_corr_centered.transpose(2, 1).contiguous())
        R = []

        for i in range(src.size(0)):
            try:
                u, s, v = torch.svd(H[i])
                r = torch.matmul(v, u.transpose(1, 0)).contiguous()
                r_det = torch.det(r).item()
                diag = torch.from_numpy(np.array([[1.0, 0, 0],
                                                  [0, 1.0, 0],
                                                  [0, 0, r_det]]).astype('float32')).to(v.device)
                r = torch.matmul(torch.matmul(v, diag), u.transpose(1, 0)).contiguous()
            except:
                print('SVD Error!')
                r = torch.eye(3)
            R.append(r)

        R = torch.stack(R, dim=0).cuda()
        t = torch.matmul(-R, src2.mean(dim=2, keepdim=True)) + src_corr2.mean(dim=2, keepdim=True)
        # t = src_corr2.mean(dim=2, keepdim=True) - src2.mean(dim=2, keepdim=True)
        if squeeze:
            return R.squeeze(0), t.squeeze(0)
        return R, t


def compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor):
    """Compute rigid transforms between two point sets
    Args:
        a (torch.Tensor): (B, M, 3) points
        b (torch.Tensor): (B, N, 3) points
        weights (torch.Tensor): (B, M)
    Returns:
        Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
    """
    assert len(a.shape) == len(b.shape)
    squeeze = False
    if len(a.shape) == 2:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
        weights = weights.unsqueeze(0)
        squeeze = True
    if a.shape[1] == 3:
        a = a.permute(0, 2, 1)
    if b.shape[1] == 3:
        b = b.permute(0, 2, 1)
    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + 1e-5)
    centroid_a = torch.sum(a * weights_normalized, dim=1)
    centroid_b = torch.sum(b * weights_normalized, dim=1)
    a_centered = a - centroid_a[:, None, :]
    b_centered = b - centroid_b[:, None, :]
    cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(rot_mat) > 0)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

    # transform = torch.cat((rot_mat, translation), dim=2)
    if squeeze:
        return rot_mat.squeeze(0), translation.squeeze(0)
    return rot_mat, translation


def sinkhorn(log_alpha, n_iters: int = 100, slack: bool = True, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1
    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """
    # Sinkhorn iterations

    squeeze = False
    if len(log_alpha.shape) == 2:
        log_alpha = log_alpha.unsqueeze(0)
        squeeze = True
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        # log_alpha = log_alpha_padded[:, :-1, :-1]
        log_alpha = log_alpha_padded
        if squeeze:
            log_alpha = log_alpha.squeeze(0)
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha

