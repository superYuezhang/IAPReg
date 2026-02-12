import torch
import numpy as np
import open3d as o3d
from typing import Union, Tuple


def get_transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    r"""Get rigid transform matrix from rotation matrix and translation vector.

    Args:
        rotation (array): (3, 3)
        translation (array): (3,)

    Returns:
        transform: (4, 4)
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def to_o3d_pcd(xyz, colors=None, normals=None):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd

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


def to_tsfm(rot, trans):
    tsfm = np.eye(4)
    tsfm[:3, :3] = rot
    tsfm[:3, 3] = trans.flatten()
    return tsfm


def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None):
    src_pcd.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)
    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])

    correspondences = np.array(correspondences)
    correspondences = torch.from_numpy(correspondences)
    return correspondences


def compute_overlap(src: Union[np.ndarray, o3d.geometry.PointCloud],
                    tgt: Union[np.ndarray, o3d.geometry.PointCloud],
                    search_voxel_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes region of overlap between two point clouds.

    Args:
        src: Source point cloud, either a numpy array of shape (N, 3) or
          Open3D PointCloud object
        tgt: Target point cloud similar to src.
        search_voxel_size: Search radius

    Returns:
        has_corr_src: Whether each source point is in the overlap region
        has_corr_tgt: Whether each target point is in the overlap region
        src_tgt_corr: Indices of source to target correspondences
    """

    if isinstance(src, np.ndarray):
        src_pcd = to_o3d_pcd(src)
        src_xyz = src
    else:
        src_pcd = src
        src_xyz = np.asarray(src.points)

    if isinstance(tgt, np.ndarray):
        tgt_pcd = to_o3d_pcd(tgt)
        tgt_xyz = tgt
    else:
        tgt_pcd = tgt
        tgt_xyz = tgt.points

    # Check which points in tgt has a correspondence (i.e. point nearby) in the src,
    # and then in the other direction. As long there's a point nearby, it's
    # considered to be in the overlap region. For correspondences, we require a stronger
    # condition of being mutual matches
    tgt_corr = np.full(tgt_xyz.shape[0], -1)
    pcd_tree = o3d.geometry.KDTreeFlann(src_pcd)
    for i, t in enumerate(tgt_xyz):
        num_knn, knn_indices, knn_dist = pcd_tree.search_radius_vector_3d(t, search_voxel_size)
        if num_knn > 0:
            tgt_corr[i] = knn_indices[0]
    src_corr = np.full(src_xyz.shape[0], -1)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)
    for i, s in enumerate(src_xyz):
        num_knn, knn_indices, knn_dist = pcd_tree.search_radius_vector_3d(s, search_voxel_size)
        if num_knn > 0:
            src_corr[i] = knn_indices[0]

    # Compute mutual correspondences
    src_corr_is_mutual = np.logical_and(tgt_corr[src_corr] == np.arange(len(src_corr)),
                                        src_corr > 0)
    src_tgt_corr = np.stack([np.nonzero(src_corr_is_mutual)[0],
                             src_corr[src_corr_is_mutual]])

    has_corr_src = src_corr >= 0
    has_corr_tgt = tgt_corr >= 0

    return has_corr_src, has_corr_tgt, src_tgt_corr


def se3_init(rot, trans):
    # if rot is None:
    #     return
    pose = np.concatenate([rot, trans], axis=-1)
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


def se3_transform(pose, xyz):
    """Apply rigid transformation to points

    Args:
        pose: ([B,] 3, 4)
        xyz: ([B,] N, 3)

    Returns:

    """

    assert xyz.shape[-1] == 3 and pose.shape[:-2] == xyz.shape[:-2]

    rot, trans = pose[..., :3, :3], pose[..., :3, 3:4]
    transformed = np.einsum('...ij,...bj->...bi', rot, xyz) + trans.transpose(-1, -2)  # Rx + t

    return transformed