from typing import List

import torch
import torch.nn.functional as F
import numpy as np


def quat2mat_torch(quat, eps=0.0):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: [B, 4]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    assert quat.ndim == 2 and quat.shape[1] == 4, quat.shape
    norm_quat = quat.norm(p=2, dim=1, keepdim=True)
    norm_quat = quat / (norm_quat + eps)
    qw, qx, qy, qz = (
        norm_quat[:, 0],
        norm_quat[:, 1],
        norm_quat[:, 2],
        norm_quat[:, 3],
    )
    B = quat.size(0)

    s = 2.0  # * Nq = qw*qw + qx*qx + qy*qy + qz*qz
    X = qx * s
    Y = qy * s
    Z = qz * s
    wX = qw * X
    wY = qw * Y
    wZ = qw * Z
    xX = qx * X
    xY = qx * Y
    xZ = qx * Z
    yY = qy * Y
    yZ = qy * Z
    zZ = qz * Z
    rotMat = torch.stack(
        [
            1.0 - (yY + zZ),
            xY - wZ,
            xZ + wY,
            xY + wZ,
            1.0 - (xX + zZ),
            yZ - wX,
            xZ - wY,
            yZ + wX,
            1.0 - (xX + yY),
        ],
        dim=1,
    ).reshape(B, 3, 3)
    return rotMat

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def allo_to_ego_mat_torch(translation, rot_allo, eps=1e-4):
    """
    Args:
        translation: Nx3
        rot_allo: Nx3x3
    """
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = torch.tensor(
        [0, 0, 1.0], dtype=translation.dtype, device=translation.device
    )  # (3,)
    obj_ray = translation / (torch.norm(translation, dim=1, keepdim=True) + eps)

    # cam_ray.dot(obj_ray), assume cam_ray: (0, 0, 1)
    angle = obj_ray[:, 2:3].acos()

    # Compute rotation between ray to object centroid and optical center ray
    axis = torch.cross(cam_ray.expand_as(obj_ray), obj_ray)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + eps)

    # Build quaternion representing the rotation around the computed axis
    # angle-axis => quat
    q_allo_to_ego = cat(
        [
            torch.cos(angle / 2.0),
            axis[:, 0:1] * torch.sin(angle / 2.0),
            axis[:, 1:2] * torch.sin(angle / 2.0),
            axis[:, 2:3] * torch.sin(angle / 2.0),
        ],
        dim=1,
    )
    rot_allo_to_ego = quat2mat_torch(q_allo_to_ego)
    # Apply quaternion for transformation from allocentric to egocentric.
    rot_ego = torch.matmul(rot_allo_to_ego, rot_allo)
    return rot_ego

import cv2
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def preprocess(image, bbox, verbose=False):
        preprocess_compose = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        roi_img, square_bbox = crop_square_resize(image, bbox, crop_size=224, interpolation=cv2.INTER_CUBIC)
        roi_img = roi_img.astype(np.uint8)
        if verbose:
            plt.imshow(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')  # Turn off axis labels
            plt.show()
        roi_img = Image.fromarray(roi_img)
        roi_img = preprocess_compose(roi_img)


        shape = image.shape
        bbox_normalized = [
            bbox[0] / shape[1],  # normalized center_x
            bbox[1] / shape[0],  # normalized center_y
            bbox[2] / shape[1],  # normalized size_x
            bbox[3] / shape[0]   # normalized size_y
        ]
        if verbose:
            print(f'The shape of the image is {shape}, the we consider the bbox (cx,cy,w,h): {bbox}, normalized as : {bbox_normalized}')
        return roi_img.unsqueeze(0), bbox_normalized

from bbx import xywh2xyxy
def crop_square_resize(
    img: np.ndarray, bbox: int, crop_size: int = None, interpolation=None
) -> np.ndarray:
    """
    Crop and resize an image to a square of size crop_size, centered on the given bounding box.

    Args:
    -----------
    img : numpy.ndarray
        Input image to be cropped and resized.
    bbox : int
        Bounding box coordinates of the object of interest. Must be in the format x1, y1, x2, y2.
    crop_size : int
        The size of the output square image. Default is None, which will use the largest dimension of the bbox as the crop_size.
    interpolation : int, optional
        The interpolation method to use when resizing the image. Default is None, which will use cv2.INTER_LINEAR.

    Returns:
    --------
    numpy.ndarray
        The cropped and resized square image.

    Raises:
    -------
    ValueError:
        If crop_size is not an integer.
    """

    if not isinstance(crop_size, int):
        raise ValueError("crop_size must be an int")

    x1, y1, x2, y2 = xywh2xyxy(bbox)
    bw = round(bbox[2])  # Bbox[2]
    bh = round(bbox[3]) # Bbox[3]
    bbox_center = bbox[0],bbox[1]

    if bh > bw:
        x1 = bbox_center[0] - bh / 2
        x2 = bbox_center[0] + bh / 2
    else:
        y1 = bbox_center[1] - bw / 2
        y2 = bbox_center[1] + bw / 2

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    if img.ndim > 2:
        roi_img = np.zeros((max(bh, bw), max(bh, bw), img.shape[2]), dtype=np.uint8)
    else:
        roi_img = np.zeros((max(bh, bw), max(bh, bw)), dtype=np.uint8)
    roi_x1 = max((0 - x1), 0)
    x1 = max(x1, 0)
    roi_x2 = roi_x1 + min((img.shape[1] - x1), (x2 - x1))
    roi_y1 = max((0 - y1), 0)
    y1 = max(y1, 0)
    roi_y2 = roi_y1 + min((img.shape[0] - y1), (y2 - y1))
    x2 = min(x2, img.shape[1])
    y2 = min(y2, img.shape[0])

    try:
        # Ajustement dynamique
        src_region = img[y1:y2, x1:x2].copy()
        dst_region = roi_img[roi_y1:roi_y2, roi_x1:roi_x2]
        
        h, w = min(src_region.shape[0], dst_region.shape[0]), min(src_region.shape[1], dst_region.shape[1])
        roi_img[roi_y1:roi_y1 + h, roi_x1:roi_x1 + w] = src_region[:h, :w]
    except ValueError as e:
        print(f"Error during broadcasting: {e}")
        print(f"Source region: {src_region.shape}, Target region: {dst_region.shape}")
        print(f"bbox: {bbox}")
        print(f"roi_y1: {roi_y1}, roi_y2: {roi_y2}, roi_x1: {roi_x1}, roi_x2: {roi_x2}")
        print(f"y1: {y1}, y2: {y2}, x1: {x1}, x2: {x2}")
        raise
    if roi_img.shape[0] == 0 or roi_img.shape[1] == 0:
        raise ValueError("roi_img is None")
    roi_img = cv2.resize(roi_img, (crop_size, crop_size), interpolation=interpolation)
    return roi_img, [x1, y1, x2, y2]