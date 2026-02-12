import torch
import numpy as np
import torch.nn as nn
from torch_scatter import scatter

from .tools import knn_point, euler2mat
import torch.nn.functional as F
from timm.models.resnet import BasicBlock, Bottleneck


class PCProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_size = 8
        self.trans_dim = 8
        self.graph_dim = 64
        self.imgblock_dim = 64
        self.img_size = 224
        self.obj_size = 224
        self.img_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.img_std = torch.Tensor([0.229, 0.224, 0.225])

        self.input_trans = nn.Conv1d(3, self.trans_dim, 1)
        self.graph_layer = nn.Sequential(nn.Conv2d(self.trans_dim * 2, self.graph_dim, kernel_size=1, bias=False),
                                         nn.GroupNorm(4, self.graph_dim),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.proj_layer = nn.Conv1d(self.graph_dim, self.graph_dim, kernel_size=1)

        self.img_block = nn.Sequential(
            BasicBlock(self.graph_dim, self.graph_dim),
            nn.Conv2d(self.graph_dim, self.graph_dim, kernel_size=1),
        )
        self.img_layer = nn.Conv2d(self.graph_dim, 3, kernel_size=1)

        self.offset = torch.Tensor([[-1, -1], [-1, 0], [-1, 1],
                                    [0, -1], [0, 0], [0, 1],
                                    [1, -1], [1, 0], [1, 1]])

        self.img_offset = torch.Tensor([[-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2],
                                        [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
                                        [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
                                        [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
                                        [2, -2], [2, -1], [2, 0], [2, 1], [2, 2]])
        # multi-view
        TRANS = 0
        _views = np.asarray([
            [[0, 0, 0], [0, 0, TRANS]],
            [[0, 1 * np.pi / 2, 0], [0, 0, TRANS]],
            [[0, 2 * np.pi / 2, 0], [0, 0, TRANS]],
            [[0, 3 * np.pi / 2, 0], [0, 0, TRANS]],
            [[1 * np.pi / 2, 0, 0], [0, 0, TRANS]],
            [[-1 * np.pi / 2, 0, 0], [0, 0, TRANS]]]).astype('float32')
        self.num_views = _views.shape[0]

        angle = torch.tensor(_views[:, 0, :]).float()
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        self.translation = torch.tensor(_views[:, 1, :]).float()
        self.translation = self.translation.unsqueeze(1)

    def get_mvpc(self, points):
        squeeze = False
        if len(points.shape) == 2:
            squeeze = True
            points = points.unsqueeze(0)
        if points.shape[1] == 3:
            points = points.permute(0, 2, 1)
        b, N, C = points.shape
        v = self.translation.shape[0]

        _points = self.point_transform(
            points=torch.repeat_interleave(points, v, dim=0),
            rot_mat=self.rot_mat.repeat(b, 1, 1),
            translation=self.translation.repeat(b, 1, 1))
        _points = _points.reshape(b, 6, N, C)
        # if squeeze:
        #     _points = _points.squeeze(0)
        return _points

    @staticmethod
    def point_transform(points, rot_mat, translation):
        """
        :param points: [batch, num_points, 3]
        :param rot_mat: [batch, 3]
        :param translation: [batch, 1, 3]
        :return:
        """
        rot_mat = rot_mat.to(points.device)
        translation = translation.to(points.device)
        points = torch.matmul(points, rot_mat)
        points = points - translation
        return points

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k, k):
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            idx = knn_point(k, coor_k.transpose(1, 2).contiguous(), coor_q.transpose(1, 2).contiguous())  # B G k
            idx = idx.transpose(1, 2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def I2P(self, pc, img_f, pc_min, grid_size, offsets, view='xy'):
        B, N, _ = pc.shape

        if view == 'xy':
            view_idx = [0, 1]
        elif view == 'yx':
            view_idx = [1, 0]
        elif view == 'yz':
            view_idx = [1, 2]
        elif view == 'zy':
            view_idx = [2, 1]
        elif view == 'xz':
            view_idx = [0, 2]
        elif view == 'zx':
            view_idx = [2, 0]
        else:
            raise "view must in [xy, yz, yz, zy, xz, zx]."

        # Point Index
        idx_xy = torch.floor((pc[:, :, view_idx] - pc_min) / grid_size)  # B N 2

        # Point Densify
        idx_xy_dense = idx_xy + 1

        # Object to Image Center
        idx_xy_dense_offset = idx_xy_dense + torch.cat(offsets, dim=1).unsqueeze(dim=1)  # B, N, 2

        # Expand Point Features
        B, C, H, W = img_f.shape
        f_dense = F.interpolate(img_f, size=(self.img_size, self.img_size), mode='bicubic').reshape(B, C, -1).permute(0, 2,
                                                                                                                  1)  # B, N, C
        # B N 9 C -> B 9N C
        assert idx_xy_dense_offset.min() >= 0 and idx_xy_dense_offset.max() <= (self.img_size - 1), str(
            idx_xy_dense_offset.min()) + '-' + str(idx_xy_dense_offset.max())

        # change idx to 1-dim
        new_idx_xy_dense = idx_xy_dense_offset[:, :, 0] * self.img_size + idx_xy_dense_offset[:, :, 1]  # B, N

        # Get Image Features
        out = torch.gather(f_dense, 1, new_idx_xy_dense.to(dtype=torch.int64).unsqueeze(-1).repeat(1, 1, C))
        return out

    def proj2color(self, original_pc, pc=None):
        if pc is None:
            pc = original_pc
        B, N, _ = pc.shape

        # calculate range
        pc_range = pc.max(dim=1)[0] - pc.min(dim=1)[0]  # B 3
        grid_size = pc_range[:, :2].max(dim=-1)[0] / (self.obj_size - 3)  # B,
        # Point Index
        pc_min = pc.min(dim=1)[0][:, :2].unsqueeze(dim=1)
        grid_size = grid_size.unsqueeze(dim=1).unsqueeze(dim=2)
        idx_xy = torch.floor((pc[:, :, :2] - pc_min) / grid_size)  # B N 2
        # 点数扩大为9倍, 增加点密度
        idx_xy_dense = (idx_xy.unsqueeze(dim=2) + self.offset.unsqueeze(dim=0).unsqueeze(dim=0).to(pc.device)).view(
            idx_xy.size(0), N * 9, 2) + 1
        # xy移动到图像的中心, B 9*N 2
        idx_xy_dense_center = torch.floor((idx_xy_dense.max(dim=1)[0] + idx_xy_dense.min(dim=1)[0]) / 2).int()
        offset_x = self.obj_size / 2 - idx_xy_dense_center[:, 0:1] - 1
        offset_y = self.obj_size / 2 - idx_xy_dense_center[:, 1:2] - 1
        idx_xy_dense_offset = idx_xy_dense + torch.cat([offset_x, offset_y], dim=1).unsqueeze(dim=1)

        # Get features
        original_pc = original_pc.transpose(1, 2).contiguous()  # B, 3, N
        f = self.input_trans(original_pc)
        f = self.get_graph_feature(original_pc, f, original_pc, f, self.local_size)
        f = self.graph_layer(f)
        f = f.max(dim=-1, keepdim=False)[0]  # B C N

        f = self.proj_layer(f).transpose(1, 2).contiguous()  # B N C
        f_dense = f.unsqueeze(dim=2).expand(-1, -1, 9, -1).contiguous().view(f.size(0), N * 9, self.graph_dim)  # B 9N C

        idx_zero = idx_xy_dense_offset < 0
        idx_obj = idx_xy_dense_offset > 223
        idx_xy_dense_offset = idx_xy_dense_offset + idx_zero.to(torch.int32)
        idx_xy_dense_offset = idx_xy_dense_offset - idx_obj.to(torch.int32)

        # change idx to 1-dim
        new_idx_xy_dense = idx_xy_dense_offset[:, :, 0] * self.obj_size + idx_xy_dense_offset[:, :, 1]
        # scatter the features, new_idx_xy_dense中相同的元素对应的f_dense求和作为新的元素
        out = scatter(f_dense, new_idx_xy_dense.long(), dim=1, reduce="sum")
        # need to pad
        if out.size(1) < self.obj_size * self.obj_size:
            delta = self.obj_size * self.obj_size - out.size(1)
            zero_pad = torch.zeros(out.size(0), delta, out.size(2)).to(out.device)
            res = torch.cat([out, zero_pad], dim=1).reshape((out.size(0), self.obj_size, self.obj_size, out.size(2)))  # B 224 224 C
        else:
            res = out.reshape((out.size(0), self.obj_size, self.obj_size, out.size(2)))  # B 224 224 C

        img_feat = self.img_block(res.permute(0, 3, 1, 2).contiguous())
        img = self.img_layer(img_feat)  # B 3 224 224
        mean_vec = self.imagenet_mean.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).to(img.device)  # 1 3 1 1
        std_vec = self.imagenet_std.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).to(img.device)  # 1 3 1 1
        # Normalize the pic
        img = nn.Sigmoid()(img)
        img_norm = img.sub(mean_vec).div(std_vec)  # (B, 3, 224, 224)

        return img_norm, pc_min, grid_size, (offset_x, offset_y)

    def proj2depth(self, pc, view='xy'):
        B, N, _ = pc.shape

        # calculate range
        pc_range = pc.max(dim=1)[0] - pc.min(dim=1)[0]  # B 3
        if view == 'xy':
            view_idx = [0, 1]
        elif view == 'yx':
            view_idx = [1, 0]
        elif view == 'yz':
            view_idx = [1, 2]
        elif view == 'zy':
            view_idx = [2, 1]
        elif view == 'xz':
            view_idx = [0, 2]
        elif view == 'zx':
            view_idx = [2, 0]
        else:
            raise "view must in [xy, yz, yz, zy, xz, zx]."

        grid_size = pc_range[:, view_idx].max(dim=-1)[0] / (self.img_size - 3)  # B,
        # Point Index
        pc_min = pc.min(dim=1)[0][:, view_idx].unsqueeze(dim=1)
        grid_size = grid_size.unsqueeze(dim=1).unsqueeze(dim=2)
        idx_xy = torch.floor((pc[:, :, view_idx] - pc_min) / grid_size)  # B N 2

        # Point Densify, (B N 2)
        idx_xy_dense = (idx_xy.unsqueeze(dim=2) + self.img_offset.unsqueeze(dim=0).unsqueeze(dim=0).to(pc.device)).view(
            idx_xy.size(0), N * 25, 2) + 1

        # Object to Image Center
        idx_xy_dense_center = torch.floor((idx_xy_dense.max(dim=1)[0] + idx_xy_dense.min(dim=1)[0]) / 2).int()
        offset_x = self.img_size / 2 - idx_xy_dense_center[:, 0: 1] - 1
        offset_y = self.img_size / 2 - idx_xy_dense_center[:, 1: 2] - 1
        idx_xy_dense_offset = idx_xy_dense + torch.cat([offset_x, offset_y], dim=1).unsqueeze(dim=1)

        # Expand Point Features
        f_dense = pc.unsqueeze(dim=2).expand(-1, -1, 25, -1).contiguous().view(pc.size(0), N * 25, 3)[..., 2: 3].repeat(
            1, 1, 3)

        idx_zero = idx_xy_dense_offset < 0
        idx_obj = idx_xy_dense_offset > 223
        idx_xy_dense_offset = idx_xy_dense_offset + idx_zero.to(torch.int32)
        idx_xy_dense_offset = idx_xy_dense_offset - idx_obj.to(torch.int32)

        # B N 9 C -> B 9N C
        assert idx_xy_dense_offset.min() >= 0 and idx_xy_dense_offset.max() <= (self.img_size - 1), str(
            idx_xy_dense_offset.min()) + '-' + str(idx_xy_dense_offset.max())

        # change idx to 1-dim
        new_idx_xy_dense = idx_xy_dense_offset[:, :, 0] * self.img_size + idx_xy_dense_offset[:, :, 1]

        # Get Image Features
        out = scatter(f_dense, new_idx_xy_dense.long(), dim=1, reduce='sum')

        # need to pad
        if out.size(1) < self.img_size * self.img_size:
            delta = self.img_size * self.img_size - out.size(1)
            zero_pad = torch.zeros(out.size(0), delta, out.size(2)).to(out.device)
            res = torch.cat([out, zero_pad], dim=1).reshape((out.size(0), self.img_size, self.img_size, out.size(2)))
        else:
            res = out.reshape((out.size(0), self.img_size, self.img_size, out.size(2)))

            # B 224 224 C
        img = res.permute(0, 3, 1, 2).contiguous()
        mean_vec = self.img_mean.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).to(pc.device)  # 1 3 1 1
        std_vec = self.img_std.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).to(pc.device)  # 1 3 1 1
        img_bin = torch.gt(img, 0) + 0
        # Normalize the pic
        img_sig = nn.Sigmoid()(img)
        img_norm = img_sig.sub(mean_vec).div(std_vec)
        # print(img_norm[0, 0, 112, ...])
        # (B, 3, 224, 224)
        return img_bin, img, img_norm, pc_min, grid_size, (offset_x, offset_y)


if __name__ == '__main__':
    pc = torch.rand(10, 100, 3)
    proj = PCProj()
    img, img_norm, _, _, _ = proj.proj2depth(pc)
    print(img.shape)


