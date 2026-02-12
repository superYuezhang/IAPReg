import torch
import torch.nn as nn
import torch.nn.functional as F
from .Project.project import PCProj
from .pretrain.model import VisionTransformer


class MVFeature(nn.Module):
    def __init__(self, cfg):
        super(MVFeature, self).__init__()
        self.proj = PCProj()
        self.img_emb = VisionTransformer(output_dim=cfg.d_embed, layers=cfg.vit_layers)
        # self.img_emb, _ = load_vit()
        self.linear = nn.Sequential(
            nn.Linear(cfg.d_embed * 6, cfg.d_embed),
            # nn.ReLU(),
            # nn.Linear(cfg.d_embed * 2, cfg.d_embed)
        )

    def get_mv_img(self, pc):
        if not isinstance(pc, torch.Tensor):
            imgs = []
            for i in range(len(pc)):
                mv_i = self.proj.get_mvpc(pc[i])
                img1_bin, img1_ori, img1, pc_min1, grid_size1, offsets1 = self.proj.proj2depth(mv_i[:, 0, ...])  # (1, 3, 224, 224)
                img2_bin, img2_ori, img2, pc_min2, grid_size2, offsets2 = self.proj.proj2depth(mv_i[:, 1, ...])
                img3_bin, img3_ori, img3, pc_min3, grid_size3, offsets3 = self.proj.proj2depth(mv_i[:, 2, ...])
                img4_bin, img4_ori, img4, pc_min4, grid_size4, offsets4 = self.proj.proj2depth(mv_i[:, 3, ...])
                img5_bin, img5_ori, img5, pc_min5, grid_size5, offsets5 = self.proj.proj2depth(mv_i[:, 4, ...])
                img6_bin, img6_ori, img6, pc_min6, grid_size6, offsets6 = self.proj.proj2depth(mv_i[:, 5, ...])
                # imgs = torch.stack([img1, img2, img3, img4, img5, img6], dim=1)  # (1, 6, 3, 224, 224)
                # mv_imgs.append(imgs)
                img_list = [img1, img2, img3, img4, img5, img6]
                img_bin_list = [img1_bin, img2_bin, img3_bin, img4_bin, img5_bin, img6_bin]
                img_bin = torch.zeros_like(img1_bin)  # init to view 1, (3, 224, 224)
                img = torch.zeros_like(img1)

                for j in range(len(img_bin_list)):
                    if torch.sum(img_bin_list[j]) >= torch.sum(img_bin):
                        img_bin = img_bin_list[j]
                        img = img_list[j]
                imgs.append(img)
        else:
            mv_pc = self.proj.get_mvpc(pc)
            img1_bin, img1_ori, img1, pc_min1, grid_size1, offsets1 = self.proj.proj2depth(mv_pc[:, 0, ...])  # (B, 3, 224, 224)
            img2_bin, img2_ori, img2, pc_min2, grid_size2, offsets2 = self.proj.proj2depth(mv_pc[:, 1, ...])
            img3_bin, img3_ori, img3, pc_min3, grid_size3, offsets3 = self.proj.proj2depth(mv_pc[:, 2, ...])
            img4_bin, img4_ori, img4, pc_min4, grid_size4, offsets4 = self.proj.proj2depth(mv_pc[:, 3, ...])
            img5_bin, img5_ori, img5, pc_min5, grid_size5, offsets5 = self.proj.proj2depth(mv_pc[:, 4, ...])
            img6_bin, img6_ori, img6, pc_min6, grid_size6, offsets6 = self.proj.proj2depth(mv_pc[:, 5, ...])
            # mv_imgs = torch.stack([img1, img2, img3, img4, img5, img6], dim=1)  # (B, 6, 3, 224, 224)
            img_list = [img1, img2, img3, img4, img5, img6]
            img_bin_list = [img1_bin, img2_bin, img3_bin, img4_bin, img5_bin, img6_bin]
            img_bin = torch.zeros_like(img1_bin)
            imgs = torch.zeros_like(img1)
            for i in range(pc.shape[0]):
                img_bin[i] = img1_bin[i]  # init to view 1, (3, 224, 224)
                imgs[i] = img1[i]
                for j in range(len(img_bin_list)):
                    if torch.sum(img_bin_list[j][i]) >= torch.sum(img_bin[i]):
                        img_bin[i] = img_bin_list[j][i]
                        imgs[i] = img_list[j][i]

        return imgs

    def MView(self, pc):
        if pc.shape[1] == 3:
            pc = pc.permute(0, 2, 1)

        img1_bin, img1_ori, img1, pc_min1, grid_size1, offsets1 = self.proj.proj2depth(pc, view='xy')  # (1, 3, 224, 224)
        img2_bin, img2_ori, img2, pc_min2, grid_size2, offsets2 = self.proj.proj2depth(pc, view='yx')
        img3_bin, img3_ori, img3, pc_min3, grid_size3, offsets3 = self.proj.proj2depth(pc, view='yz')
        img4_bin, img4_ori, img4, pc_min4, grid_size4, offsets4 = self.proj.proj2depth(pc, view='zy')
        img5_bin, img5_ori, img5, pc_min5, grid_size5, offsets5 = self.proj.proj2depth(pc, view='xz')
        img6_bin, img6_ori, img6, pc_min6, grid_size6, offsets6 = self.proj.proj2depth(pc, view='zx')
        imgs = torch.cat((img1, img2, img3, img4, img5, img6), dim=0)  # B*6, 3, 224, 224

        img_feat, img_sali = self.img_emb(imgs)

        img_feat = img_feat.float()
        img_sali = img_sali.float()[:, 0, 1:]  # B*6, 196
        B = img_sali.shape[0]

        # Multi view image feature
        img_feat1 = img_feat[0: B // 6]
        img_feat2 = img_feat[B // 6 * 1: B // 6 * 2]
        img_feat3 = img_feat[B // 6 * 2: B // 6 * 3]
        img_feat4 = img_feat[B // 6 * 3: B // 6 * 4]
        img_feat5 = img_feat[B // 6 * 4: B // 6 * 5]
        img_feat6 = img_feat[B // 6 * 5:]
        pc_feat1 = self.proj.I2P(pc, img_feat1, pc_min1, grid_size1, offsets1, view='xy')
        pc_feat2 = self.proj.I2P(pc, img_feat2, pc_min2, grid_size2, offsets2, view='yx')
        pc_feat3 = self.proj.I2P(pc, img_feat3, pc_min3, grid_size3, offsets3, view='yz')
        pc_feat4 = self.proj.I2P(pc, img_feat4, pc_min4, grid_size4, offsets4, view='zy')
        pc_feat5 = self.proj.I2P(pc, img_feat5, pc_min5, grid_size5, offsets5, view='xz')
        pc_feat6 = self.proj.I2P(pc, img_feat6, pc_min6, grid_size6, offsets6, view='zx')
        pc_feat = torch.cat([pc_feat1, pc_feat2, pc_feat3, pc_feat4, pc_feat5, pc_feat6], dim=-1)  # [B, N, C*6]
        pc_feat = self.linear(pc_feat)  # [B, N, C]

        # Image Attention(Saliency) Maps
        img_sali = img_sali.reshape(-1, 1, 14, 14)  # B, 1, 14, 14
        img_sali1 = img_sali[: B // 6]
        img_sali2 = img_sali[B // 6 * 1: B // 6 * 2]
        img_sali3 = img_sali[B // 6 * 2: B // 6 * 3]
        img_sali4 = img_sali[B // 6 * 3: B // 6 * 4]
        img_sali5 = img_sali[B // 6 * 4: B // 6 * 5]
        img_sali6 = img_sali[B // 6 * 5:]
        # back-proj img to point cloud, B, N
        pc_sali1 = self.proj.I2P(pc, img_sali1, pc_min1, grid_size1, offsets1, view='xy').squeeze(-1)
        pc_sali2 = self.proj.I2P(pc, img_sali2, pc_min2, grid_size2, offsets2, view='yx').squeeze(-1)
        pc_sali3 = self.proj.I2P(pc, img_sali3, pc_min3, grid_size3, offsets3, view='yz').squeeze(-1)
        pc_sali4 = self.proj.I2P(pc, img_sali4, pc_min4, grid_size4, offsets4, view='zy').squeeze(-1)
        pc_sali5 = self.proj.I2P(pc, img_sali5, pc_min5, grid_size5, offsets5, view='xz').squeeze(-1)
        pc_sali6 = self.proj.I2P(pc, img_sali6, pc_min6, grid_size6, offsets6, view='zx').squeeze(-1)
        pc_sali = (pc_sali1 + pc_sali2 + pc_sali3 + pc_sali4 + pc_sali5 + pc_sali6) / 6
        # B, N
        pc_sali = F.softmax(pc_sali, dim=1)
        pc_sali = pc_sali.unsqueeze(-1)  # B, N, 1

        return pc_feat, pc_sali
    
    def forward(self, pc):
        pass


