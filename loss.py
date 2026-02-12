import torch
import torch.nn as nn
import torch.nn.functional as F


class LossL1(nn.Module):
    def __init__(self):
        super(LossL1, self).__init__()
        self.loss = nn.L1Loss()

    def __call__(self, input, target):
        return self.loss(input, target)


class LossL2(nn.Module):
    def __init__(self):
        super(LossL2, self).__init__()
        self.loss = nn.MSELoss()

    def __call__(self, input, target):
        return self.loss(input, target)


class InfoNCELossFull(nn.Module):
    """Computes InfoNCE loss
    """
    def __init__(self, d_embed, r_p, r_n):
        """
        Args:
            d_embed: Embedding dimension
            r_p: Positive radius (points nearer than r_p are matches)
            r_n: Negative radius (points nearer than r_p are not matches)
        """
        super().__init__()
        self.r_p = r_p
        self.r_n = r_n
        self.n_sample = 256
        self.W = torch.nn.Parameter(torch.zeros(d_embed, d_embed), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.W, std=0.1)

    def compute_infonce(self, anchor_feat, positive_feat, anchor_xyz, positive_xyz):
        """
        Args:
            anchor_feat: Shape ([B,] N_anc, D)
            positive_feat: Shape ([B,] N_pos, D)
            anchor_xyz: ([B,] N_anc, 3)
            positive_xyz: ([B,] N_pos, 3)
        """
        W_triu = torch.triu(self.W)
        W_symmetrical = W_triu + W_triu.T
        match_logits = torch.einsum('...ic,cd,...jd->...ij', anchor_feat, W_symmetrical, positive_feat)  # (..., N_anc, N_pos)

        with torch.no_grad():
            dist_keypts = torch.cdist(anchor_xyz, positive_xyz)
            dist1, idx1 = dist_keypts.topk(k=1, dim=-1, largest=False)  # Finds the positive (closest match)
            mask = dist1[..., 0] < self.r_p  # Only consider points with correspondences (..., N_anc)
            ignore = dist_keypts < self.r_n  # Ignore all the points within a certain boundary,
            ignore.scatter_(-1, idx1, 0)     # except the positive (..., N_anc, N_pos)

        match_logits[..., ignore] = -float('inf')

        loss = -torch.gather(match_logits, -1, idx1).squeeze(-1) + torch.logsumexp(match_logits, dim=-1)
        loss = torch.sum(loss[mask]) / torch.sum(mask)
        return loss

    def forward(self, src_feat, tgt_feat, src_xyz, tgt_xyz):
        """
        Args:
            src_feat: List(B) of source features (N_src, D)
            tgt_feat: List(B) of target features (N_tgt, D)
            src_xyz:  List(B) of source coordinates (N_src, 3)
            tgt_xyz: List(B) of target coordinates (N_tgt, 3)

        Returns:
        """

        B = len(src_feat)
        infonce_loss = [self.compute_infonce(src_feat[b], tgt_feat[b], src_xyz[b], tgt_xyz[b]) for b in range(B)]

        return torch.mean(torch.stack(infonce_loss))


def compute_feature_loss(all_R_feats, all_t_feats):
    # enc_R_feats, rotated_R_feats, translated_R_feats
    # enc_t_feats, rotated_t_feats, translated_t_feats
    l2_criterion = LossL2()
    # R feats loss
    R_loss = 0
    t_loss = 0
    for i in range(len(all_R_feats)):

        R_feats_pos = l2_criterion(all_t_feats[i][0], all_t_feats[i][1])
        R_feats_neg = l2_criterion(all_R_feats[i][0], all_R_feats[i][1])
        R_loss_i = (torch.clamp(-R_feats_neg + 0.01, min=0.0) +
                                                R_feats_pos) * 0.0005
        R_loss += R_loss_i
        # t feats loss
        t_feats_pos = l2_criterion(all_R_feats[i][0], all_R_feats[i][2])
        t_feats_neg = l2_criterion(all_t_feats[i][0], all_t_feats[i][2])
        t_loss_i = (torch.clamp(-t_feats_neg + 0.01, min=0.0) +
                                                t_feats_pos) * 0.0005
        t_loss += t_loss_i

    return R_loss + t_loss


def chamfer_loss(a, b):
    """ return Chamfer distance
    Args:
        a: B, N, C
        b: B, N, C
    Returns:
    """
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    # diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
    diag_ind = torch.arange(0, num_points)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return torch.mean(torch.min(P, 1)[0]) + torch.mean(torch.min(P, 2)[0])


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        # CHANGED torch.sum -> torch.mean
        loss_1 = torch.mean(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.mean(mins)
        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        #  [batch, n, 3]
        x = x.float()
        y = y.float()
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P


class CorrLoss(nn.Module):
    def __init__(self, metric='mae'):
        super(CorrLoss, self).__init__()
        self.metric = metric

    def forward(self, src_corr_list, src_corr_list_gt, olp_score_list=None):
        corr_err = torch.cat(src_corr_list, dim=0) - torch.cat(src_corr_list_gt, dim=0)

        if self.metric == 'mae':
            corr_err = torch.sum(torch.abs(corr_err), dim=-1)  # (N, )
        elif self.metric == 'mse':
            corr_err = torch.sum(torch.square(corr_err), dim=-1)
        if olp_score_list is not None:
            overlap_weights = torch.cat(olp_score_list)
            mean_err = torch.sum(overlap_weights * corr_err) / torch.clamp_min(torch.sum(overlap_weights), 1e-6)
        else:
            mean_err = torch.mean(corr_err, dim=0)

        return mean_err


def square_distance(src, dst, normalised = False):
    """
    Calculate Euclid distance between each two points.
    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    if(normalised):
        dist += 2
    else:
        dist += torch.sum(src ** 2, dim=-1)[:, :, None]
        dist += torch.sum(dst ** 2, dim=-1)[:, None, :]

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist


class CircleLoss(nn.Module):
    def __init__(self, cfg):
        super(CircleLoss, self).__init__()
        self.safe_radius = cfg.safe_radius
        self.pos_radius = cfg.pos_radius

    def get_circle_loss(self, coords_dist, feats_dist):
        pos_mask = coords_dist < self.pos_radius
        neg_mask = coords_dist > self.safe_radius

        ## get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1) > 0) * (neg_mask.sum(-1) > 0)).detach()
        col_sel = ((pos_mask.sum(-2) > 0) * (neg_mask.sum(-2) > 0)).detach()

        # get alpha for both positive and negative pairs
        pos_weight = feats_dist - 1e5 * (~pos_mask).float()  # mask the non-positive
        pos_weight = (pos_weight - self.pos_optimal)  # mask the uninformative positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight).detach()

        neg_weight = feats_dist + 1e5 * (~neg_mask).float()  # mask the non-negative
        neg_weight = (self.neg_optimal - neg_weight)  # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight).detach()

        lse_pos_row = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight, dim=-1)
        lse_pos_col = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight, dim=-2)

        lse_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight, dim=-1)
        lse_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight, dim=-2)

        loss_row = F.softplus(lse_pos_row + lse_neg_row) / self.log_scale
        loss_col = F.softplus(lse_pos_col + lse_neg_col) / self.log_scale

        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2

        return circle_loss

    def forward(self, src_pcd, tgt_pcd, src_feats, tgt_feats, R_gt, t_gt):
        """
        Circle loss for metric learning, here we feed the positive pairs only
        Input:
            src_pcd:        [N, 3]
            tgt_pcd:        [M, 3]
            rot:            [3, 3]
            trans:          [3, 1]
            src_feats:      [N, C]
            tgt_feats:      [M, C]
        """
        trans_src = torch.matmul(R_gt, src_pcd.permute(1, 0)) + t_gt  # (3, N)
        trans_src = trans_src.permute(1, 0)  # (N, 3)
        coords_dist = torch.sqrt(square_distance(trans_src.unsqueeze(0), tgt_pcd.unsqueeze(0)).squeeze(0))
        feats_dist = torch.sqrt(square_distance(src_feats.unsqueeze(0), tgt_feats.unsqueeze(0), normalised=True)).squeeze(0)
        loss_circle = self.get_circle_loss(coords_dist, feats_dist)

        return loss_circle
