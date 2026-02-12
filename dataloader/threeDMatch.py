import os, sys, glob, h5py, pickle, random
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from .transforms import *
from .data_Util import compute_overlap, se3_init, se3_transform, se3_inv, to_o3d_pcd


def collate_pair(list_data):
    """Collates data using a list, for tensors which are of different sizes
    (e.g. different number of points). Otherwise, stacks them as per normal.
    """

    batch_sz = len(list_data)

    # Collate as normal, other than fields that cannot be collated due to differing sizes,
    # we retain it as a python list
    to_retain_as_list = ['src_xyz', 'tgt_xyz', 'tgt_raw',
                         'src_overlap', 'tgt_overlap',
                         'correspondences',
                         'src_path', 'tgt_path',
                         'idx']
    data = {k: [list_data[b][k] for b in range(batch_sz)] for k in to_retain_as_list if k in list_data[0]}
    data['pose'] = torch.stack([list_data[b]['pose'] for b in range(batch_sz)], dim=0)  # (B, 3, 4)
    data['R'] = torch.stack([list_data[b]['R'] for b in range(batch_sz)], dim=0)  # (B, 3, 3)
    data['t'] = torch.stack([list_data[b]['t'] for b in range(batch_sz)], dim=0)  # (B, 3, )
    if 'overlap_p' in list_data[0]:
        data['overlap_p'] = torch.tensor([list_data[b]['overlap_p'] for b in range(batch_sz)])
    return data


def get_3DLomatch_datasets(cfg, phase='train'):
    dataset = ThreeDMatch(cfg, phase=phase, data_type='3DLoMatch', max_points=cfg.max_points)

    return dataset


def get_3Dmatch_datasets(cfg, phase='train'):
    dataset = ThreeDMatch(cfg, phase=phase, data_type='3DMatch', max_points=cfg.max_points)

    return dataset


class ThreeDMatch(Dataset):
    def __init__(self, cfg, phase, data_type='3DMatch', noise=False, max_points=4096):
        assert phase in ['train', 'val', 'test']
        assert data_type in ['3DMatch', '3DLoMatch']
        self.data_dir = os.path.join(cfg.root, '3DMatch')
        self.info_dir = os.path.join(self.data_dir, 'infos')
        self.noise = noise
        transforms_aug = None
        if phase in ['train']:
            info_fname = os.path.join(self.info_dir, f'{phase}_info.pkl')
            # info_fname = os.path.join(self.info_dir, f'{phase}_info1_reduce_samp.pkl')
            # info_fname = os.path.join(self.info_dir, f'{phase}_info.pkl')
            # Apply training data augmentation (Pose perturbation and jittering)
            # transforms_aug = torchvision.transforms.Compose([
            #     RigidPerturb(perturb_mode='small'),
            #     Jitter(scale=0.005),
            #     ShufflePoints(),
            #     RandomSwap(),
            # ])
        elif phase in ['val']:
            # info_fname = os.path.join(self.info_dir, f'{phase}_info1_reduce_samp.pkl')
            info_fname = os.path.join(self.info_dir, f'{phase}_info.pkl')
        elif phase in ['test']:
            # info_fname = os.path.join(self.info_dir, f'{phase}_{data_type}_info1_reduce_samp.pkl')
            info_fname = os.path.join(self.info_dir, f'{phase}_{data_type}_info.pkl')

        with open(info_fname, 'rb') as fid:
            self.infos = pickle.load(fid)

        self.overlap_radius = cfg.overlap_radius
        self.phase = phase
        self.max_points = max_points
        self.transforms_aug = transforms_aug

    def __getitem__(self, item):
        rot = self.infos['rot'][item]
        trans = self.infos['trans'][item]
        pose = se3_init(rot, trans)  # transforms src to tgt (3, 4)
        # get pointcloud
        src_path = os.path.join(self.data_dir, self.infos['src'][item])
        tgt_path = os.path.join(self.data_dir, self.infos['tgt'][item])
        src_pcd = torch.load(src_path)
        tgt_pcd = torch.load(tgt_path)
        src_pcd = src_pcd.astype('float32')
        tgt_pcd = tgt_pcd.astype('float32')

        # vds_rate = src_pcd.shape[0] / 10000 * 0.03  # 越大点数越少
        # pcd = to_o3d_pcd(src_pcd)
        # pcd_down = pcd.voxel_down_sample(vds_rate)
        # src_pcd = np.asarray(pcd_down.points).astype(np.float32)
        # pcd = to_o3d_pcd(tgt_pcd)
        # pcd_down = pcd.voxel_down_sample(vds_rate)
        # tgt_pcd = np.asarray(pcd_down.points).astype(np.float32)
        # # print(src_pcd.shape, tgt_pcd.shape)
        if (src_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(src_pcd.shape[0])[:self.max_points]
            src_pcd = src_pcd[idx]
        if (tgt_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(tgt_pcd.shape[0])[:self.max_points]
            tgt_pcd = tgt_pcd[idx]

        src_mask_gt, tgt_mask_gt, src_tgt_corr = compute_overlap(
            se3_transform(pose, src_pcd),
            tgt_pcd,
            self.overlap_radius,
        )

        src_pcd = torch.from_numpy(src_pcd).float()  # (N, 3)
        tgt_pcd = torch.from_numpy(tgt_pcd).float()
        rot = torch.from_numpy(rot.astype('float32'))
        trans = torch.from_numpy(trans.astype('float32'))
        trans = trans.reshape(3, 1)  # (3, )
        # transform_gt = torch.cat([rot, trans], dim=1)
        datas = {
            'src_path': src_path,
            'tgt_path': tgt_path,
            'pose': pose.astype('float32'),
            'src_xyz': src_pcd,
            'tgt_xyz': tgt_pcd,
            'src_overlap': torch.from_numpy(src_mask_gt),  # 部分重叠点云的重叠掩码
            'tgt_overlap': torch.from_numpy(tgt_mask_gt),
            # 'transformer': transform_gt.float(),
            'R': rot,
            't': trans.squeeze(),
        }
        # if self.transforms_aug is not None:
        #     self.transforms_aug(datas)  # Apply data augmentation
        datas['pose'] = torch.from_numpy(pose.astype('float32'))

        return datas

    def __len__(self):
        return len(self.infos['rot'])

