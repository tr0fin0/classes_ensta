import math
from scipy import spatial
import numpy as np
from scipy.linalg import logm
import numpy.linalg as LA
import torch


#    def proj(pts, pose_pred, pose_gt, K):
#        """
#        average re-projection error in 2d
#
#        :param pts: nx3 ndarray with 3D model points.
#        :param pose_pred: Estimated pose (3x3 rot. matrix and 3x1 translation vector).
#        :param pose_gt: GT pose (3x3 rot. matrix and 3x1 translation vector).
#        :param K: Camera intrinsics to project the model onto the image plane.
#        :return:
#        """
#        rot_pred = pose_pred[:3, :3]
#        t_pred = pose_pred[:, 3]
#
#        rot_gt = pose_gt[:3, :3]
#        t_gt = pose_gt[:, 3]
#
#        proj_pred = self.project_pts(pts, rot_pred, t_pred, K)
#        proj_gt = self.project_pts(pts, rot_gt, t_gt, K)
#        e = np.linalg.norm(proj_pred - proj_gt, axis=1).mean()
#        return e
def se3_mul(RT1, RT2):
    """
    concat 2 RT transform
    :param RT1=[R,T], 4x3 np array
    :param RT2=[R,T], 4x3 np array
    :return: RT_new = RT1 * RT2
    """
    R1 = RT1[0:3, 0:3]
    T1 = RT1[0:3, 3].reshape((3, 1))

    R2 = RT2[0:3, 0:3]
    T2 = RT2[0:3, 3].reshape((3, 1))

    RT_new = np.zeros((3, 4), dtype=np.float32)
    RT_new[0:3, 0:3] = np.dot(R1, R2)
    T_new = np.dot(R1, T2) + T1
    RT_new[0:3, 3] = T_new.reshape((3))
    return RT_new


def transform_pts(pts, rot, t):
    """
    Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param rot: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert pts.shape[1] == 3
    pts_t = rot.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T


def project_pts(pts, rot, t, K):
    """
    Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param rot: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :param K: 3x3 intrinsic matrix
    :return: nx2 ndarray with transformed 2D points.
    """
    assert pts.shape[1] == 3
    if K.shape == (9,):
        K = K.reshape(3, 3)
    pts_t = rot.dot(pts.T) + t.reshape((3, 1))  # 3xn
    pts_c_t = K.dot(pts_t)
    n = pts.shape[0]
    pts_2d = np.zeros((n, 2))
    pts_2d[:, 0] = pts_c_t[0, :] / pts_c_t[2, :]
    pts_2d[:, 1] = pts_c_t[1, :] / pts_c_t[2, :]

    return pts_2d


def calc_depth_img(pts, rot, t, K, w=640, h=480):
    """
    Project 3D points onto the image plane and create a depth image by storing z at each pixel
    """
    assert pts.shape[1] == 3
    if K.shape == (9,):
        K = K.reshape(3, 3)
    pts_t = rot.dot(pts.T) + t.reshape((3, 1))  # 3xn
    pts_c_t = K.dot(pts_t)
    n = pts.shape[0]
    pts_2d = np.zeros((n, 2))
    pts_2d[:, 0] = pts_c_t[0, :] / pts_c_t[2, :]
    pts_2d[:, 1] = pts_c_t[1, :] / pts_c_t[2, :]

    pts_2d = pts_2d.astype(np.int)

    depth_img = np.zeros((h, w))

    for pt_2d, z in zip(pts_2d, pts_c_t[2, :]):
        u = pt_2d[0]
        v = pt_2d[1]
        # Check if current object point is inside image
        if u < 0 or u >= w or v < 0 or v >= h:
            continue
        # Check if the depth at current pixel is zero:
        if depth_img[v, u] == 0:
            depth_img[v, u] = z
        elif depth_img[v, u] > z:
            depth_img[v, u] = z
        else:
            continue
    # Filter image to fill black holes in the projected object
    for i, row in enumerate(depth_img):
        obj_pixels = row.nonzero()[0]
        if len(obj_pixels) == 0:
            continue
        for j in range(obj_pixels[0], obj_pixels[-1]):
            if row[j] == 0:
                # Average over surrounding pixels
                values = []
                for l in [-1, 0, 1]:
                    for k in [-1, 0, 1]:
                        if l == 0 and k == 0:
                            continue
                        if (i + l) >= h or (i + l) < 0:
                            continue
                        if (j + k) >= w or (j + k) < 0:
                            continue
                        if depth_img[i + l, j + k] == 0:
                            continue
                        values.append(depth_img[i + l, j + k])
                if len(values) != 0:
                    depth_img[i, j] = sum(values) / len(values)
    return depth_img


def calc_add(pts, pose_pred, pose_gt):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).
    http://www.stefan-hinterstoisser.com/papers/hinterstoisser2012accv.pdf

    :param pts: nx3 ndarray with 3D model points.
    :param pose_pred: Estimated pose (3x3 rot. matrix and 3x1 translation vector).
    :param pose_gt: GT pose (3x3 rot. matrix and 3x1 translation vector).
    :return: Mean average error between the predicted and ground truth pose.
    """
    rot_pred = pose_pred[:3, :3]
    t_pred = pose_pred[:, 3]

    rot_gt = pose_gt[:3, :3]
    t_gt = pose_gt[:, 3]

    pts_est = transform_pts(pts, rot_pred, t_pred)
    pts_gt = transform_pts(pts, rot_gt, t_gt)
    error = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return error


def calc_adi(pts, pose_pred, pose_gt):
    """
    Average Distance of Model Points for objects with indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).
    http://www.stefan-hinterstoisser.com/papers/hinterstoisser2012accv.pdf

    :param pts: nx3 ndarray with 3D model points.
    :param pose_pred: Estimated pose (3x3 rot. matrix and 3x1 translation vector).
    :param pose_gt: GT pose (3x3 rot. matrix and 3x1 translation vector).
    :return: Mean average error between the predicted and ground truth pose reduced by symmetry.
    """
    rot_pred = pose_pred[:3, :3]
    t_pred = pose_pred[:, 3]

    rot_gt = pose_gt[:3, :3]
    t_gt = pose_gt[:, 3]

    pts_pred = transform_pts(pts, rot_pred, t_pred)
    pts_gt = transform_pts(pts, rot_gt, t_gt)

    # Calculate distances to the nearest neighbors from pts_gt to pts_est
    nn_index = spatial.cKDTree(pts_pred)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    error = nn_dists.mean()
    return error


def calc_rotation_error(rot_pred, r_gt):
    """
    Calculate the angular geodesic rotation error between a predicted rotation matrix and the ground truth matrix.
    :paran rot_pred: Predicted rotation matrix (3x3)
    :param rot_gt: Ground truth rotation matrix (3x3)
    """
    assert rot_pred.shape == r_gt.shape == (3, 3)
    temp = logm(np.dot(np.transpose(rot_pred), r_gt))
    rd_rad = LA.norm(temp, "fro") / np.sqrt(2)
    rd_deg = rd_rad / np.pi * 180
    return rd_deg


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(
            b2_x1
        )  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center dist ** 2
            if (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * (
                    torch.atan(w2 / h2) - torch.atan(w1 / h1)
                ).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return (
            iou - (c_area - union) / c_area
        )  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU
