import os.path as osp
import random, pickle
import numpy as np
import torch.utils.data
from .data_Util import compute_overlap, se3_init, se3_transform, to_o3d_pcd



def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


class KittiDataset(torch.utils.data.Dataset):
    ODOMETRY_KITTI_DATA_SPLIT = {
        'train': ['00', '01', '02', '03', '04', '05'],
        'val': ['06', '07'],
        'test': ['08', '09', '10'],
    }

    def __init__(
        self,
        cfg,
        dataset_root='/home/zy/dataset/KITTI_odometry',
        subset='train',
        point_limit=None,
    ):
        super(KittiDataset, self).__init__()

        self.dataset_root = dataset_root
        self.subset = subset
        self.point_limit = point_limit
        self.max_points = cfg.max_points

        self.metadata = load_pickle(osp.join(self.dataset_root, 'metadata', f'{subset}.pkl'))

    def _load_point_cloud(self, file_name):
        points = np.load(file_name)
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points

    def __getitem__(self, index):
        data_dict = {}

        metadata = self.metadata[index]
        data_dict['seq_id'] = metadata['seq_id']
        data_dict['ref_frame'] = metadata['frame0']
        data_dict['src_frame'] = metadata['frame1']
        # print(metadata.keys())

        # src_points = self._load_point_cloud(osp.join(self.dataset_root, metadata['pcd0']))
        # ref_points = self._load_point_cloud(osp.join(self.dataset_root, metadata['pcd1']))

        src_points = self._load_point_cloud(osp.join(self.dataset_root, metadata['pcd1']))
        ref_points = self._load_point_cloud(osp.join(self.dataset_root, metadata['pcd0']))

        vds_rate = src_points.shape[0] / 10000 * 0.8  # 越大点数越少
        pcd = to_o3d_pcd(src_points)
        pcd_down = pcd.voxel_down_sample(vds_rate)
        src_points = np.asarray(pcd_down.points).astype(np.float32)
        pcd = to_o3d_pcd(ref_points)
        pcd_down = pcd.voxel_down_sample(vds_rate)
        ref_points = np.asarray(pcd_down.points).astype(np.float32)
        # print(src_points.shape, ref_points.shape)
        if (src_points.shape[0] > self.max_points):
            idx = np.random.permutation(src_points.shape[0])[:self.max_points]
            src_points = src_points[idx]

        if (ref_points.shape[0] > self.max_points):
            idx = np.random.permutation(ref_points.shape[0])[:self.max_points]
            ref_points = ref_points[idx]
        transform = metadata['transform'].astype('float32')
        rot = transform[:3, :3]
        trans = transform[:3, 3].reshape(3, 1)
        # print(rot.shape, trans.shape)
        pose = se3_init(rot, trans)  # transforms src to tgt
        src_mask_gt, tgt_mask_gt, src_tgt_corr = compute_overlap(
            se3_transform(pose, src_points),
            ref_points,
            0.375,
        )

        data_dict = {
            'src_overlap': src_mask_gt,  # 部分重叠点云的重叠掩码
            'tgt_overlap': tgt_mask_gt,
            'pose': pose.astype('float32'),
            'R': rot,
            't': trans.squeeze(),
        }
        data_dict['src_xyz'] = ref_points.astype(np.float32)
        data_dict['tgt_xyz'] = src_points.astype(np.float32)
        # data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        # data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transformation'] = transform.astype(np.float32)

        return data_dict

    def __len__(self):
        return len(self.metadata)

