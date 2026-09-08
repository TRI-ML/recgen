import numpy as np
import trimesh
import open3d as o3d

from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree

from .geometry import np_transform_pcd
from .utils import points_to_open3d, homogeneous


def compute_RT_distances(pose1: np.ndarray, pose2: np.ndarray):
    '''
    :param RT_1: [B, 4, 4]. homogeneous affine transformation
    :param RT_2: [B, 4, 4]. homogeneous affine transformation
    :return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter
    Works in batched or unbatched manner. NB: assumes that translations are in Meters
    '''

    if pose1 is None or pose2 is None:
        return -1

    if len(pose1.shape) == 2:
        pose1 = np.expand_dims(pose1, axis=0)
        pose2 = np.expand_dims(pose2, axis=0)

    try:
        assert np.array_equal(pose1[:, 3, :], pose2[:, 3, :])
        assert np.array_equal(pose1[0, 3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        import warnings
        warnings.warn("compute_RT_distances: pose bottom row is not [0,0,0,1]")

    R1 = pose1[:, :3, :3] / np.cbrt(np.linalg.det(pose1[:, :3, :3]))[:, None, None]
    T1 = pose1[:, :3, 3]

    R2 = pose2[:, :3, :3] / np.cbrt(np.linalg.det(pose2[:, :3, :3]))[:, None, None]
    T2 = pose2[:, :3, 3]

    R = np.matmul(R1, R2.transpose(0, 2, 1))
    arccos_arg = (np.trace(R, axis1=1, axis2=2) - 1) / 2
    arccos_arg = np.clip(arccos_arg, -1 + 1e-12, 1 - 1e-12)
    theta = np.arccos(arccos_arg) * 180 / np.pi
    theta[np.isnan(theta)] = 180.
    shift = np.linalg.norm(T1 - T2, axis=-1) * 100

    return theta, shift


def compute_add(pcd: np.ndarray, pred_pose: np.ndarray, gt_pose: np.ndarray) -> np.ndarray:
    pred_r, pred_t = pred_pose[:3, :3], pred_pose[:3, 3]
    gt_r, gt_t = gt_pose[:3, :3], gt_pose[:3, 3]

    model_pred = np_transform_pcd(pcd, pred_r, pred_t)
    model_gt = np_transform_pcd(pcd, gt_r, gt_t)

    # ADD computation
    add = np.mean(np.linalg.norm(model_pred - model_gt, axis=1))

    return add


def compute_adds(pcd: np.ndarray, pred_pose: np.ndarray, gt_pose: np.ndarray) -> np.ndarray:
    pred_r, pred_t = pred_pose[:3, :3], pred_pose[:3, 3]
    gt_r, gt_t = gt_pose[:3, :3], gt_pose[:3, 3]

    model_pred = np_transform_pcd(pcd, pred_r, pred_t)
    model_gt = np_transform_pcd(pcd, gt_r, gt_t)

    # ADD-S computation
    kdt = KDTree(model_gt, metric='euclidean')
    distance, _ = kdt.query(model_pred, k=1)
    adds = np.mean(distance)

    return adds

def match_point_counts(pcd1, pcd2):
    n1, n2 = len(pcd1), len(pcd2)

    # Determine which is larger
    if n1 > n2:
        idx = np.random.randint(0, n1, n2)
        pcd1 = pcd1[idx]
    elif n2 > n1:
        idx = np.random.randint(0, n2, n1)
        pcd2 = pcd2[idx]

    return pcd1, pcd2

def compute_add_gen(pcd_pred: np.ndarray, pcd_gt:np.ndarray, pred_pose: np.ndarray, gt_pose: np.ndarray) -> np.ndarray:
    pred_r, pred_t = pred_pose[:3, :3], pred_pose[:3, 3]
    gt_r, gt_t = gt_pose[:3, :3], gt_pose[:3, 3]

    model_pred = np_transform_pcd(pcd_pred, pred_r, pred_t)
    model_gt = np_transform_pcd(pcd_gt, gt_r, gt_t)

    # Make sure both have the same number of points
    model_pred, model_gt = match_point_counts(model_pred, model_gt)

    # ADD computation
    add = np.mean(np.linalg.norm(model_pred - model_gt, axis=1))

    return add


def compute_adds_gen(pcd_pred: np.ndarray, pcd_gt:np.ndarray,  pred_pose: np.ndarray, gt_pose: np.ndarray) -> np.ndarray:
    pred_r, pred_t = pred_pose[:3, :3], pred_pose[:3, 3]
    gt_r, gt_t = gt_pose[:3, :3], gt_pose[:3, 3]

    model_pred = np_transform_pcd(pcd_pred, pred_r, pred_t)
    model_gt = np_transform_pcd(pcd_gt, gt_r, gt_t)

    # ADD-S computation
    kdt = KDTree(model_gt, metric='euclidean')
    distance, _ = kdt.query(model_pred, k=1)
    adds = np.mean(distance)

    return adds

def chamfer_distance_gt_mesh(gt_pose, gt_mesh, pred_pose, pred_mesh, thres=0.02, use_icp=True, return_transform=False, with_scaling=False):
    gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, 99999, face_weight=None, sample_color=False)
    pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, 99999, face_weight=None, sample_color=False)

    # Transform predicted points to GT frame using poses
    transformed_pred_pts = np_transform_pcd(pred_pts, pred_pose[:3, :3], pred_pose[:3, 3])
    inverted_gt_pose = np.linalg.inv(gt_pose)
    transformed_pred_pts = np_transform_pcd(transformed_pred_pts, inverted_gt_pose[:3, :3], inverted_gt_pose[:3, 3])

    if use_icp:
        pcd_pred = points_to_open3d(transformed_pred_pts)
        pcd_pred = pcd_pred.voxel_down_sample(0.005)
        pcd_gt = points_to_open3d(gt_pts)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=with_scaling)
        reg_p2p = o3d.pipelines.registration.registration_icp(pcd_pred, pcd_gt, thres, np.eye(4), estimation)
        icp_tf = reg_p2p.transformation
        final_pts = (icp_tf @ homogeneous(transformed_pred_pts).T).T[:, :3]
    else:
        icp_tf = np.eye(4)
        final_pts = transformed_pred_pts

    chamfer_dists = chamfer_distance_between_clouds(final_pts, gt_pts)
    chamfer_dis = chamfer_dists.mean() * 100

    if return_transform:
        return chamfer_dis, icp_tf
    return chamfer_dis

def _sample_surface_with_colors(mesh, num_points):
    """Sample surface points + RGB colors from a mesh.

    Returns:
        pts: (N, 3) float64 positions
        colors: (N, 3) uint8 RGB values
    """
    pts, _, colors = trimesh.sample.sample_surface(mesh, num_points, sample_color=True)
    return pts, colors[:, :3]


def chamfer_distance_gt_mesh_color(gt_pose, gt_mesh, pred_pose, pred_mesh, thres=0.02, icp_transform=None):
    """Compute color-aware Chamfer distance before and after ICP alignment.

    Args:
        icp_transform: Optional pre-computed 4x4 ICP transform (e.g. from chamfer_distance_gt_mesh).
                       If None, ICP is run internally.

    Returns:
        (chamfer_after_icp, chamfer_before_icp): both multiplied by 100 (cm units).
    """
    # Sample points + colors
    gt_pts, gt_colors = _sample_surface_with_colors(gt_mesh, 99999)
    pred_pts, pred_colors = _sample_surface_with_colors(pred_mesh, 99999)

    # Apply predicted pose and align into GT frame
    transformed_pred_pts = np_transform_pcd(pred_pts, pred_pose[:3, :3], pred_pose[:3, 3])
    inverted_gt_pose = np.linalg.inv(gt_pose)
    transformed_pred_pts = np_transform_pcd(transformed_pred_pts, inverted_gt_pose[:3, :3], inverted_gt_pose[:3, 3])

    # Color Chamfer before ICP
    chamfer_before = chamfer_distance_with_color(
        transformed_pred_pts, pred_colors[:, :3],
        gt_pts, gt_colors[:, :3],
        pos_weight=1.0,
        color_weight=0.05
    )

    # Determine ICP transform
    if icp_transform is None:
        pcd_pred = points_to_open3d(transformed_pred_pts)
        pcd_pred = pcd_pred.voxel_down_sample(0.005)
        pcd_gt = points_to_open3d(gt_pts)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_pred, pcd_gt, thres, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        icp_transform = reg_p2p.transformation

    pred_pts_icp = (icp_transform @ homogeneous(transformed_pred_pts).T).T[:, :3]

    # Color Chamfer after ICP
    chamfer_after = chamfer_distance_with_color(
        pred_pts_icp, pred_colors[:, :3],
        gt_pts, gt_colors[:, :3],
        pos_weight=1.0,
        color_weight=0.05
    )

    return chamfer_after * 100, chamfer_before * 100


def chamfer_distance_between_clouds(pts1,pts2):
    kdtree1 = cKDTree(pts1)
    dists1, indices1 = kdtree1.query(pts2)
    kdtree2 = cKDTree(pts2)
    dists2, indices2 = kdtree2.query(pts1)
    chamfer_dist = 0.5*(dists1.mean()+dists2.mean())   #!NOTE should not be mean of all, see https://pdal.io/en/stable/apps/chamfer.html
    return chamfer_dist



def chamfer_distance_with_color(pts1, colors1, pts2, colors2, pos_weight=1.0, color_weight=0.05):
    # Normalize color to 0-1
    colors1 = colors1 / 255.0
    colors2 = colors2 / 255.0

    # Build feature vectors combining position + color
    feat1 = np.hstack([pos_weight * pts1, color_weight * colors1])
    feat2 = np.hstack([pos_weight * pts2, color_weight * colors2])

    kdtree1 = cKDTree(feat1)
    d1, _ = kdtree1.query(feat2)

    kdtree2 = cKDTree(feat2)
    d2, _ = kdtree2.query(feat1)

    return 0.5 * (d1.mean() + d2.mean())
