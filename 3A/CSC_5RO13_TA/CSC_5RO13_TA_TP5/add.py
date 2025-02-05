from typing import Tuple

import numpy as np
from scipy import spatial
from scipy.linalg import logm
import torch

def transform_pts_Rt(pts, R, t):
    """Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert pts.shape[1] == 3
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T

def transform_pts_Rt_batch(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Applies a rigid transformation to 3D points.

    Args:
        pts (np.ndarray): Bxnx3 ndarray with B batches of n 3D points.
        R (np.ndarray): Bx3x3 ndarray with B batches of rotation matrices.
        t (np.ndarray): Bx3 ndarray with B batches of translation vectors.

    Returns:
        np.ndarray: Bxnx3 ndarray with B batches of n transformed 3D points.
    """
    assert pts.shape[-1] == 3
    B, n = pts.shape[:-1]
    assert R.shape == (B, 3, 3)
    assert t.shape == (B, 3)
    pts = pts.reshape(B,-1, 3)
    R = R.reshape(-1, 3, 3)
    t = t.reshape(-1, 3, 1)
    pts_t = np.matmul(R, pts.transpose(0,2,1)) + t
    pts_t = pts_t.reshape(*pts.shape)
    return pts_t

def add(R_est, t_est, R_gt, t_gt, pts):
    """Average Distance of Model Points for objects with no indistinguishable.

    views - by Hinterstoisser et al. (ACCV'12).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e

def add_batch(R_est: np.ndarray, t_est: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Average Distance of Model Points for objects with no indistinguishable views 
    - by Hinterstoisser et al. (ACCV'12).

    Args:
        R_est (np.ndarray): Bx3x3 ndarray with B batches of estimated rotation matrices.
        t_est (np.ndarray): Bx3 ndarray with B batches of estimated translation vectors.
        R_gt (np.ndarray): Bx3x3 ndarray with B batches of ground-truth rotation matrices.
        t_gt (np.ndarray): Bx3 ndarray with B batches of ground-truth translation vectors.
        pts (np.ndarray): Bxnx3 ndarray with B batches of n 3D model points.

    Returns:
        np.ndarray: B ndarray with the calculated errors for each batch.
    """
    pts_est = transform_pts_Rt_batch(pts, R_est, t_est)
    pts_gt = transform_pts_Rt_batch(pts, R_gt, t_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=-1).mean(axis=-1)
    return e


    
def adi(R_est, t_est, R_gt, t_gt, pts):
    """
    Average Distance of Model Points for objects with indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)

    # Calculate distances to the nearest neighbors from pts_gt to pts_est
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    e = nn_dists.mean()
    return e


def adi_batch(R_est: np.ndarray, t_est: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Average Distance of Model Points for objects with indistinguishable views - by Hinterstoisser et al. (ACCV 2012).

    Args:
        R_est (np.ndarray): Bx3x3 ndarray with B batches of estimated rotation matrices.
        t_est (np.ndarray): Bx3 ndarray with B batches of estimated translation vectors.
        R_gt (np.ndarray): Bx3x3 ndarray with B batches of ground-truth rotation matrices.
        t_gt (np.ndarray): Bx3 ndarray with B batches of ground-truth translation vectors.
        pts (np.ndarray): Bxnx3 ndarray with B batches of n 3D model points.

    Returns:
        np.ndarray: B ndarray with the calculated errors for each batch.
    """
    assert R_est.shape == t_est.shape == R_gt.shape == t_gt.shape
    B = R_est.shape[0]
    e = np.zeros(B)
    pts_est = transform_pts_Rt_batch(pts, R_est, t_est)
    pts_gt = transform_pts_Rt_batch(pts, R_gt, t_gt)
    for i in range(B):
        nn_index = spatial.cKDTree(pts_est[i])
        nn_dists, _ = nn_index.query(pts_gt[i], k=1)
        e[i] = nn_dists.mean()
    return e


def calc_rotation_error(rot_pred, r_gt):
    """
    Calculate the angular geodesic rotation error between a predicted rotation matrix and the ground truth matrix.
    :paran rot_pred: Predicted rotation matrix (3x3)
    :param rot_gt: Ground truth rotation matrix (3x3)
    """
    assert (rot_pred.shape == r_gt.shape == (3, 3))
    temp = logm(np.dot(np.transpose(rot_pred), r_gt), disp=True) #logm result may be inaccurate, approximate err = 7.362797496929104e-07
    rd_rad = np.linalg.norm(temp, 'fro') / np.sqrt(2)
    rd_deg = rd_rad / np.pi * 180
    return rd_deg


def calc_rotation_error_batch(pred_rot: np.ndarray, gt_rot: np.ndarray) -> np.ndarray:
    """
    Compute the angular loss between predicted and ground truth batched rotation matrices.
    Args:
        pred_rot (np.ndarray): The predicted batch of 3x3 rotation matrices with shape (batch_size, 3, 3).
        gt_rot (np.ndarray): The ground truth batch of 3x3 rotation matrices with shape (batch_size, 3, 3).
    Returns:
        np.ndarray: The mean angular loss.
    """
    rot_mul = np.matmul(gt_rot, np.transpose(pred_rot, (0, 2, 1)))
    cos_angle = (np.einsum("bii->b", rot_mul) - 1) / 2
    cos_angle = np.clip(cos_angle, -1, 1)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg



def calc_euler_angle_error_batch(pred_rot, gt_rot, seq='xyz'): # seq = 'xyz' for yaw, pitch, roll, I checked it on the image sparow_test_test_test_test
    """
    Calculate Euler angle error between predicted and ground truth batched rotation matrices.
    Args:
        pred_rot (np.ndarray): The predicted batch of 3x3 rotation matrices with shape (batch_size, 3, 3).
        gt_rot (np.ndarray): The ground truth batch of 3x3 rotation matrices with shape (batch_size, 3, 3).
        seq (str): Sequence of Euler angles (default: 'zyx' for yaw, pitch, and roll).
    Returns:
        np.ndarray: The mean Euler angle error for each sample in the batch.
    """
    batch_size = pred_rot.shape[0]
    euler_error = [[],[],[]]
    if isinstance(pred_rot, torch.Tensor):
        pred_rot = pred_rot.cpu().numpy()
    if isinstance(gt_rot, torch.Tensor):
        gt_rot = gt_rot.cpu().numpy()
    for i in range(batch_size):
        pred_r = spatial.transform.Rotation.from_matrix(pred_rot[i])
        gt_r = spatial.transform.Rotation.from_matrix(gt_rot[i])
        pred_r = pred_r.as_euler(seq, True)#*180/np.pi
        gt_r = gt_r.as_euler(seq, True)#*180/np.pi
        error = pred_r - gt_r
        if seq == 'xyz':
            euler_error[0].append(error[0])
            euler_error[1].append(error[1])
            euler_error[2].append(error[2])
        if seq == 'zyx':
            euler_error[0].append(error[2])
            euler_error[1].append(error[1])
            euler_error[2].append(error[0])
    return euler_error

def calc_euler_angle_error_batch_model_frame(pred_rot, gt_rot, seq='xyz'): # seq = 'xyz' for yaw, pitch, roll, I checked it on the image sparow_test_test_test_test
    batch_size = pred_rot.shape[0]
    euler_error = [[],[],[]]
    if isinstance(pred_rot, torch.Tensor):
        pred_rot = pred_rot.cpu().numpy()
    if isinstance(gt_rot, torch.Tensor):
        gt_rot = gt_rot.cpu().numpy()
    for i in range(batch_size):
        pred_r = spatial.transform.Rotation.from_matrix(pred_rot[i].T)
        gt_r = spatial.transform.Rotation.from_matrix(gt_rot[i].T)
        pred_r = pred_r.as_euler(seq, True)#*180/np.pi
        gt_r = gt_r.as_euler(seq, True)#*180/np.pi
        error = pred_r - gt_r
        if seq == 'xyz':
            euler_error[0].append(error[0])
            euler_error[1].append(error[1])
            euler_error[2].append(error[2])
        if seq == 'zyx':
            euler_error[0].append(error[2])
            euler_error[1].append(error[1])
            euler_error[2].append(error[0])
    return euler_error


def calc_translation_error_batch(t_est: np.ndarray, t_gt: np.ndarray) -> np.ndarray:
    """
    Translational Error.

    Args:
        t_est (np.ndarray): Bx3 ndarray with B batches of estimated translation vectors.
        t_gt (np.ndarray): Bx3 ndarray with B batches of ground-truth translation vectors.

    Returns:
        np.ndarray: B ndarray with the calculated errors for each batch.
    """
    assert t_est.shape == t_gt.shape
    assert t_est.shape[-1] == 3

    error = np.linalg.norm(t_gt - t_est, axis=-1)
    return error

def calc_translation_error_batch_non_abs(t_est: np.ndarray, t_gt: np.ndarray) -> np.ndarray:
    """
    Translational Error.

    Args:
        t_est (np.ndarray): Bx3 ndarray with B batches of estimated translation vectors.
        t_gt (np.ndarray): Bx3 ndarray with B batches of ground-truth translation vectors.

    Returns:
        np.ndarray: B ndarray with the calculated errors for each batch.
    """
    assert t_est.shape == t_gt.shape
    assert t_est.shape[-1] == 3

    mean_error_non_norm = np.mean(t_est - t_gt, axis=1, keepdims=False) # error is positive if t_est > t_gt 
    return mean_error_non_norm.reshape(-1)
