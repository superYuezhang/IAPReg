import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone_kpconv.kpconv import KPFEncoder, PreprocessorGPU
from transformer.position_embedding import PositionEmbeddingCoordsSine
from transformer.transformers import \
    TransformerCrossEncoderLayer, TransformerCrossEncoder
from utils.se3_torch import compute_rigid_transform, se3_transform_list, se3_inv
from utils.tools import compute_euclid, pad_list, unpad_list, split_src_tgt
from utils.rigid_estim import WeightedSVD, sinkhorn
from loss import CorrLoss
from ImgPC.MultiView import MVFeature


# GenerateCorr
class ComputeCorr(nn.Module):
    def __init__(self):
        super(ComputeCorr, self).__init__()

    def generate_corr(self, src, tgt, src_f, tgt_f):
        # src, tgt: (batch, n, 3)
        # embedding: (batch, C, N)
        # return (B, N, 3)
        squeeze = False
        if len(src.shape) == 2:
            src = src.unsqueeze(0)
            src_f = src_f.unsqueeze(0)
            tgt = tgt.unsqueeze(0)
            tgt_f = tgt_f.unsqueeze(0)
            squeeze = True
        if src.shape[1] == 3:
            src = src.permute(0, 2, 1)
        if tgt.shape[1] == 3:
            tgt = tgt.permute(0, 2, 1)
        simi_src = -compute_euclid(src_f, tgt_f)
        simi_tgt = -compute_euclid(tgt_f, src_f)

        # simi_src = sinkhorn(simi_src)
        # simi_tgt = sinkhorn(simi_tgt)
        # simi_src = torch.softmax(simi_src, dim=2)[..., :-1, :-1]  # 转化为概率, 所有的值之和为1
        # simi_tgt = torch.softmax(simi_tgt, dim=2)[..., :-1, :-1]

        simi_src = torch.softmax(simi_src, dim=2)  # 转化为概率, 所有的值之和为1
        simi_tgt = torch.softmax(simi_tgt, dim=2)

        src_corr = torch.matmul(simi_src, tgt)  # 加权平均tgt的全局特征作为对应点(n1, 3)
        tgt_corr = torch.matmul(simi_tgt, src)  # 加权平均src的全局特征作为对应点(n2, 3)

        if squeeze:
            return src_corr.squeeze(0), tgt_corr.squeeze(0)
        return src_corr, tgt_corr

    def forward(self, src_list, tgt_list, src_f_list, tgt_f_list):
        # src_list: list of (N1, 3)
        # src_f_list: list of (N, C)
        src_corr_list = []
        tgt_corr_list = []
        for i in range(len(src_list)):
            src_corr, tgt_corr = self.generate_corr(
                src_list[i], tgt_list[i], src_f_list[i].permute(1, 0), tgt_f_list[i].permute(1, 0))
            src_corr_list.append(src_corr)
            tgt_corr_list.append(tgt_corr)

        return src_corr_list, tgt_corr_list


class RegNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        architecture = ['simple',
                        'resnetb',
                        'resnetb',
                        'resnetb', ]

        # architecture = cfg.architecture

        self.preprocessor = PreprocessorGPU(cfg, architecture=architecture)
        self.kpf_encoder = KPFEncoder(cfg, architecture=architecture)
        self.feat_proj = nn.Linear(self.kpf_encoder.encoder_skip_dims[-1], cfg.d_embed, bias=True)
        self.pos_embed = PositionEmbeddingCoordsSine(3, cfg.d_embed, scale=1.0)
        self.img_f = MVFeature(cfg)

        # pc transformer
        encoder_layer = TransformerCrossEncoderLayer(
            cfg.d_embed, cfg.nhead, cfg.d_feedforward, cfg.dropout,
            activation=cfg.transformer_act,
            normalize_before=cfg.pre_norm,
            sa_val_has_pos_emb=True,
            ca_val_has_pos_emb=True,
            attention_type=cfg.attention_type,
        )
        encoder_norm = nn.LayerNorm(cfg.d_embed) if cfg.pre_norm else None
        # (N, B, C)
        self.transformer_encoder_pc = TransformerCrossEncoder(
            encoder_layer, cfg.num_encoder_layers, encoder_norm,
            return_intermediate=False)
        # img transformer
        encoder_layer_img = TransformerCrossEncoderLayer(
            cfg.d_embed, cfg.nhead, cfg.d_feedforward, cfg.dropout,
            activation=cfg.transformer_act,
            normalize_before=cfg.pre_norm,
            sa_val_has_pos_emb=False,
            ca_val_has_pos_emb=False,
            attention_type=cfg.attention_type,
        )
        encoder_norm_img = nn.LayerNorm(cfg.d_embed) if cfg.pre_norm else None
        self.transformer_encoder_img = TransformerCrossEncoder(
            encoder_layer_img, cfg.num_encoder_layers, encoder_norm_img,
            return_intermediate=False)

        self.corr_gener = ComputeCorr()
        self.rigid_predictor = WeightedSVD()
        self.w_loss = cfg.w_loss
        self.nn_score = nn.Linear(cfg.d_embed * 2 + 1, 1)

        self.score_src_nn = nn.Sequential(
            nn.Conv1d(cfg.d_embed * 2 + 1, 128, 1), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 1), nn.ReLU(inplace=True),
            nn.Conv1d(128, 2, 1),
        )
        self.score_tgt_nn = nn.Sequential(
            nn.Conv1d(cfg.d_embed * 2 + 1, 128, 1), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 1), nn.ReLU(inplace=True),
            nn.Conv1d(128, 2, 1),
        )

    def forward(self, datas):
        # print(datas['src_xyz'].shape)

        src_olp_list = list(datas['src_xyz'])
        tgt_olp_list = list(datas['tgt_xyz'])

        B = len(datas['src_xyz'])
        # Preprocess
        kpconv_meta = self.preprocessor(src_olp_list + tgt_olp_list)
        datas['kpconv_meta'] = kpconv_meta
        slens = [s.tolist() for s in
                 kpconv_meta['stack_lengths']]  # [[len_src,..., len_tgt,...], [采样后len_src,..., len_tgt,...]]
        slens_c = slens[-1]  # 采样后的点长度
        src_slens_c, tgt_slens_c = slens_c[:B], slens_c[B:]
        feats0 = torch.ones_like(kpconv_meta['points'][0][:, 0:1])
        # print(kpconv_meta['points'])  # list(2): 原始点和下采样后的点 [(N1, 3), (N2, 3)], N1, N2都是batch*(src+tgt的点数)

        feats_kp, dense_feats_kp = self.kpf_encoder(feats0, kpconv_meta)
        both_feats_kp = self.feat_proj(feats_kp)
        src_f_kp, tgt_f_kp = split_src_tgt(both_feats_kp, slens_c)  # list[(N_s, 256), ...]batch个元素

        src_samp_list, tgt_samp_list = split_src_tgt(kpconv_meta['points'][-1], slens_c)  # 采样后的点[(N2, 3), ...]batch个元素
        src_pe, tgt_pe = split_src_tgt(self.pos_embed(kpconv_meta['points'][-1]),
                                       slens_c)  # list[(N2, C), ...]: batch个元素
        src_pe_pad, _, _ = pad_list(src_pe)  # (batch, N2, C)
        tgt_pe_pad, _, _ = pad_list(tgt_pe)

        src_f_pad, src_f_pad_mask, src_feats_len = pad_list(src_f_kp, require_padding_mask=True)  # (B, N, C)
        tgt_f_pad, tgt_f_pad_mask, tgt_feats_len = pad_list(tgt_f_kp, require_padding_mask=True)
        src_samp_pad, src_samp_pad_mask, src_pad_len = pad_list(src_samp_list, require_padding_mask=True)
        tgt_samp_pad, tgt_samp_pad_mask, tgt_pad_len = pad_list(tgt_samp_list, require_padding_mask=True)

        # img Feature
        src_img_f, src_sali = self.img_f.MView(src_samp_pad)  # (B, N, C)
        tgt_img_f, tgt_sali = self.img_f.MView(tgt_samp_pad)

        # (6, batch, N2, C), features of each layer (6 in all)
        src_img_f, tgt_img_f = self.transformer_encoder_img(
            src_img_f.permute(1, 0, 2), tgt_img_f.permute(1, 0, 2),
            src_key_padding_mask=src_f_pad_mask,
            tgt_key_padding_mask=tgt_f_pad_mask,
        )
        src_img_f = src_img_f[-1]  # (batch, N, C)
        tgt_img_f = tgt_img_f[-1]
        # (6, batch, N2, C), features of each layer (6 in all)
        src_f, tgt_f = self.transformer_encoder_pc(
            src_f_pad.permute(1, 0, 2), tgt_f_pad.permute(1, 0, 2),
            src_key_padding_mask=src_f_pad_mask,
            tgt_key_padding_mask=tgt_f_pad_mask,
            src_pos=src_pe_pad.permute(1, 0, 2),
            tgt_pos=tgt_pe_pad.permute(1, 0, 2),
        )
        src_f = src_f[-1]  # (B, N, C)
        tgt_f = tgt_f[-1]
        src_f = torch.cat([src_f, src_img_f], dim=-1)
        tgt_f = torch.cat([tgt_f, tgt_img_f], dim=-1)
        src_f_list = unpad_list(src_f, src_feats_len)
        tgt_f_list = unpad_list(tgt_f, tgt_feats_len)

        src_corr_list, tgt_corr_list = self.corr_gener(
            src_samp_list, tgt_samp_list, src_f_list, tgt_f_list)

        src_corr_pad, src_corr_pad_mask, src_corr_len = pad_list(src_corr_list, require_padding_mask=True)
        src_corr_img_f, src_corr_sali = self.img_f.MViewAgg(src_corr_pad)
        tgt_corr_pad, tgt_corr_pad_mask, tgt_corr_len = pad_list(tgt_corr_list, require_padding_mask=True)
        tgt_corr_img_f, tgt_corr_sali = self.img_f.MViewAgg(tgt_corr_pad)

        src_score_sali = -torch.square(src_sali - src_corr_sali)  # (B, N)
        tgt_score_sali = -torch.square(tgt_sali - tgt_corr_sali)  # (B, N)
        # src_score_sali = src_score_sali / torch.min(src_score_sali)
        # tgt_score_sali = tgt_score_sali / torch.min(tgt_score_sali)

        src_f = torch.cat([src_f, src_score_sali], dim=-1)
        tgt_f = torch.cat([tgt_f, tgt_score_sali], dim=-1)
        src_mask_onehot = self.score_src_nn(src_f.permute(0, 2, 1))   # (B, 2, N)
        tgt_mask_onehot = self.score_tgt_nn(tgt_f.permute(0, 2, 1))
        src_mask_onehot_list = unpad_list(src_mask_onehot.permute(0, 2, 1), src_feats_len)  # [(N1, 2),]
        tgt_mask_onehot_list = unpad_list(tgt_mask_onehot.permute(0, 2, 1), tgt_feats_len)
        src_score = torch.max(src_mask_onehot, dim=1)[1]  # (B, N)
        tgt_score = torch.max(tgt_mask_onehot, dim=1)[1]
        src_score_list = unpad_list(src_score.unsqueeze(-1), src_feats_len)  # list of (N, 1)
        tgt_score_list = unpad_list(tgt_score.unsqueeze(-1), tgt_feats_len)

        # 防止训练初期点数太少无法SVD
        min_num = 20
        for i in range(len(src_score_list)):
            if torch.sum(torch.eq(src_score_list[i], 1)) < min_num:
                src_mask_score = torch.softmax(src_mask_onehot[i], dim=0)[1, :src_feats_len[i]]  # (N',)
                # 取前overlap_points个点作为重叠点
                src_mask = torch.zeros(src_mask_score.shape).to(src_f.device)
                values, indices = torch.topk(src_mask_score, k=min_num, dim=0)
                src_mask.scatter_(0, indices, 1)  # (dim, 索引, 根据索引赋的值)
                src_score_list[i] = src_mask.reshape(-1, 1)
            if torch.sum(torch.eq(tgt_score_list[i], 1)) < min_num:
                tgt_mask_score = torch.softmax(tgt_mask_onehot[i], dim=0)[1, :tgt_feats_len[i]]
                # 取前overlap_points个点作为重叠点
                tgt_mask = torch.zeros(tgt_mask_score.shape).to(tgt_f.device)
                values, indices = torch.topk(tgt_mask_score, k=min_num, dim=0)
                tgt_mask.scatter_(0, indices, 1)  # (dim, 索引, 根据索引赋的值)
                tgt_score_list[i] = tgt_mask.reshape(-1, 1)

        Rs1 = []
        ts1 = []
        Rs2 = []
        ts2 = []
        for i in range(len(src_corr_list)):
            R1, t1 = self.rigid_predictor(src_samp_list[i], src_corr_list[i],
                                          src_score_list[i].squeeze(-1))
            R2, t2 = self.rigid_predictor(tgt_corr_list[i], tgt_samp_list[i],
                                          tgt_score_list[i].squeeze(-1))

            Rs1.append(R1.unsqueeze(0))
            ts1.append(t1.unsqueeze(0))
            Rs2.append(R2.unsqueeze(0))
            ts2.append(t2.unsqueeze(0))
        R1 = torch.cat(Rs1, dim=0)  # (B, 3, 3)
        t1 = torch.cat(ts1, dim=0)  # (B, 3, 1)
        R2 = torch.cat(Rs1, dim=0)  # (B, 3, 3)
        t2 = torch.cat(ts1, dim=0)  # (B, 3, 1)
        transformer1 = torch.cat([R1, t1.reshape(-1, 3, 1)], dim=-1)
        transformer2 = torch.cat([R2, t2.reshape(-1, 3, 1)], dim=-1)

        trans_src_samp_list = se3_transform_list(datas['transformer'], src_samp_list)
        trans_src_samp_imgs_list = self.img_f.get_mv_img(trans_src_samp_list)  # list of (1, 6, 3, 224, 224)
        src_corr_imgs_list = self.img_f.get_mv_img(src_corr_list)

        trans_tgt_samp_list = se3_transform_list(se3_inv(datas['transformer']), tgt_samp_list)
        trans_tgt_samp_imgs_list = self.img_f.get_mv_img(trans_tgt_samp_list)  # list of (1, 6, 3, 224, 224)
        tgt_corr_imgs_list = self.img_f.get_mv_img(tgt_corr_list)

        outputs = {}
        outputs_ = {
            'trans_src_samp_imgs_list': trans_src_samp_imgs_list,
            'src_corr_imgs_list': src_corr_imgs_list,
            'trans_tgt_samp_imgs_list': trans_tgt_samp_imgs_list,
            'tgt_corr_imgs_list': tgt_corr_imgs_list,

            'src_samp_list': src_samp_list,
            'tgt_samp_list': tgt_samp_list,

            'src_corr_list': src_corr_list,
            'tgt_corr_list': tgt_corr_list,

            'src_mask_onehot_list': src_mask_onehot_list,  # [(N1, 2),]
            'tgt_mask_onehot_list': tgt_mask_onehot_list,
            'src_score_list': src_score_list,
            'tgt_score_list': tgt_score_list,

            'src_f_list': src_f_list,  # list of (N, C)
            'tgt_f_list': tgt_f_list,

            'transformer1': transformer1,
            'transformer2': transformer2,
            'R': R1,
            't': t1,
            'R2': R2,
            't2': t2,
        }
        outputs.update(outputs_)

        loss_reg = self.compute_loss(outputs, datas)
        loss = loss_reg

        return outputs, loss

    def compute_loss(self, outputs, datas):
        # Cross transformer loss
        R_pred = outputs['R']
        t_pred = outputs['t']
        batch_size = R_pred.shape[0]
        E = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(R_pred.device)

        # GT transformer loss
        R_loss = F.mse_loss(torch.matmul(R_pred, datas['R'].permute(0, 2, 1)), E)
        t_loss = F.mse_loss(t_pred.reshape(-1, 3), datas['t'].reshape(-1, 3))
        loss_trans = R_loss + t_loss
        # Score Loss
        loss_fn_olp = nn.CrossEntropyLoss()  # (B, 2, N), (B, N)

        src_score_gt_list = list(datas['src_overlap'])
        tgt_score_gt_list = list(datas['tgt_overlap'])
        src_mask_onehot = torch.cat(outputs['src_mask_onehot_list'], dim=0).unsqueeze(0).permute(0, 2, 1)
        tgt_mask_onehot = torch.cat(outputs['tgt_mask_onehot_list'], dim=0).unsqueeze(0).permute(0, 2, 1)
        # src_mask_onehot = outputs['src_mask_onehot'].permute(0, 2, 1)
        # tgt_mask_onehot = outputs['tgt_mask_onehot'].permute(0, 2, 1)
        src_overlap_gt = torch.cat(src_score_gt_list, dim=0).squeeze(-1).unsqueeze(0).long()
        tgt_overlap_gt = torch.cat(tgt_score_gt_list, dim=0).squeeze(-1).unsqueeze(0).long()
        # src_overlap_gt = src_score_gt_list.long()
        # tgt_overlap_gt = tgt_score_gt_list.long()
        loss_score = loss_fn_olp(src_mask_onehot, src_overlap_gt) + loss_fn_olp(tgt_mask_onehot, tgt_overlap_gt)

        # Score Loss
        # loss_fn_score = nn.BCEWithLogitsLoss()
        # # src_mask_gt_list, tgt_mask_gt_list, src_score_list, tgt_score_list = compute_overlap_mask_list(
        # #     outputs['src_samp_list'], outputs['tgt_samp_list'], datas['R'], datas['t'], overlap_radius=0.01)
        # # datas['src_mask_gt_list'] = src_mask_gt_list
        # # datas['tgt_mask_gt_list'] = tgt_mask_gt_list
        # #
        # # kpconv_meta = datas['kpconv_meta']
        # # p = len(kpconv_meta['stack_lengths']) - 1  # coarsest level
        # # datas['overlap_pyr'] = compute_overlaps2(datas)
        # # src_score_gt_list, tgt_score_gt_list = \
        # #     split_src_tgt(datas['overlap_pyr'][f'pyr_{p}'], kpconv_meta['stack_lengths'][p])
        #
        # src_score_gt_list = list(datas['src_overlap'])
        # tgt_score_gt_list = list(datas['tgt_overlap'])
        #
        # src_score_gt = torch.cat(src_score_gt_list, dim=0).unsqueeze(-1).float()  # (N, 1)
        # tgt_score_gt = torch.cat(tgt_score_gt_list, dim=0).unsqueeze(-1).float()
        #
        # src_score = torch.cat(outputs['src_score_list'], dim=0)  # (N, 1)
        # tgt_score = torch.cat(outputs['tgt_score_list'], dim=0)
        # # print(src_score_gt.dtype, src_score.dtype)
        # loss_score = loss_fn_score(src_score, src_score_gt) + loss_fn_score(tgt_score, tgt_score_gt)

        # Correspondence Loss
        corr_loss_fn = CorrLoss()
        loss_corr_src = corr_loss_fn(outputs['src_corr_list'],
                                     se3_transform_list(datas['transformer'], outputs['src_samp_list']),
                                     src_score_gt_list)

        loss_corr_tgt = corr_loss_fn(outputs['tgt_corr_list'],
                                     se3_transform_list(se3_inv(datas['transformer']), outputs['tgt_samp_list']),
                                     tgt_score_gt_list)
        loss_corr = loss_corr_src + loss_corr_tgt

        # image loss
        trans_src_samp_imgs = torch.cat(outputs['trans_src_samp_imgs_list'], dim=0)
        src_corr_imgs = torch.cat(outputs['src_corr_imgs_list'], dim=0)
        loss_img_src = F.mse_loss(trans_src_samp_imgs, src_corr_imgs)
        trans_tgt_samp_imgs = torch.cat(outputs['trans_tgt_samp_imgs_list'], dim=0)
        tgt_corr_imgs = torch.cat(outputs['tgt_corr_imgs_list'], dim=0)
        loss_img_tgt = F.mse_loss(trans_tgt_samp_imgs, tgt_corr_imgs)
        loss_img = loss_img_src + loss_img_tgt

        loss = loss_trans + loss_score*0.5 + loss_corr*0.5 + loss_img
        # print(loss_trans.item(), loss_score.item(), loss_corr.item(), loss_img.item())
        return loss




