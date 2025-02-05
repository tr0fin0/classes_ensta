from pathlib import Path
import pickle as pkl
import copy

import torch
import numpy as np
from scipy.integrate import simps

import math
from scipy import spatial

# from models.detr_pose.loss.matcher import HungarianMatcher
# from models.rt_pose.loss.matcher import HungarianMatcher

from metrics import (
    calc_add,
    calc_adi,
)
from add import (
    calc_rotation_error_batch,
    calc_translation_error_batch,
    calc_translation_error_batch_non_abs,
    calc_euler_angle_error_batch,
    calc_rotation_error,
    calc_euler_angle_error_batch_model_frame,
)


class PoseMetrics:
    def __init__(
        self,
        save_dir,
        model_pts,
        class_names,
        diameters,
        binary_symmetries,
        matcher=None,
        pred_cls=False

    ):
        self.save_dir = Path(save_dir) / "pose_metrics"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model_pts = model_pts
        self.class_names = class_names
        self.diameters = diameters
        self.diameter_thresholds = {
            k: 0.1 * diameter for k, diameter in diameters.items()
        }
        self.model_symmetries = binary_symmetries
        self.matcher = matcher
        self.pred_cls = pred_cls

        self.name_img = []
        self.azimuth_angles = []
        self.elevation_angles = []
        self.roll_angles = []
        self.pitch_angles = []
        self.yaw_angles = []
        self.x_position = []
        self.y_position = []
        self.z_position = []

        self.poses_pred = {}
        self.poses_gt = {}
        self.scores_pred = {}
        self.scores_gt = {}
        self.num = {}
        self.ang_distances_cls = {}
        self.ang_euler_distances_cls = {}
        self.ang_euler_distances_model_frame_cls = {}
        self.trans_distances_cls = {}
        self.trans_distances_cls_non_abs = {}
        self.z_distances_cls = {}
        self.z_distances_cls_non_abs = {}
        self.x_distances_cls_non_abs = {}
        self.y_distances_cls_non_abs = {}
        self.add_cls = {}
        self.adds_cls = {}
        self.adds_tp_cls = {}

        self.thresholds_trans = [0.01, 0.02, 0.05]
        self.thresholds_ang = [1, 2, 5]
        self.ONEcmONEdeg_tp_cls = {}
        self.TWOcmTWOdeg_tp_cls = {}
        self.FIVEcmFIVEdeg_tp_cls = {}
        self.ONEcm_tp_cls = {}
        self.TWOcm_tp_cls = {}
        self.FIVEcm_tp_cls = {}
        self.ONEdeg_tp_cls = {}
        self.TWOdeg_tp_cls = {}
        self.FIVEdeg_tp_cls = {}

        self.reset()

    def reset(self):
        """
        Reset the PoseEvaluator stored poses. Necessary when the same evaluator is used during training
        """
        self.poses_pred = {}
        self.poses_gt = {}
        self.scored_pred = {}
        self.scores_gt = {}
        self.num = {}
        self.ang_distances_cls = {}
        self.ang_euler_distances_cls = {}
        self.ang_euler_distances_model_frame_cls = {}
        self.trans_distances_cls = {}
        self.trans_distances_cls_non_abs = {}
        self.z_distances_cls = {}
        self.z_distances_cls_non_abs = {}
        self.x_distances_cls_non_abs = {}
        self.y_distances_cls_non_abs = {}
        self.add_cls = {}
        self.adds_cls = {}
        self.adds_tp_cls = {}
        self.scores_cls = {}

        self.name_img = []
        self.azimuth_angles = []
        self.elevation_angles = []
        self.roll_angles = []
        self.pitch_angles = []
        self.yaw_angles = []
        self.x_position = []
        self.y_position = []
        self.z_position = []

        self.ONEcmONEdeg_tp_cls = {}
        self.TWOcmTWOdeg_tp_cls = {}
        self.FIVEcmFIVEdeg_tp_cls = {}
        self.ONEcm_tp_cls = {}
        self.TWOcm_tp_cls = {}
        self.FIVEcm_tp_cls = {}
        self.ONEdeg_tp_cls = {}
        self.TWOdeg_tp_cls = {}
        self.FIVEdeg_tp_cls = {}

        for cls_id in self.class_names.keys():
            self.num[cls_id] = 0
            self.poses_pred[cls_id] = []
            self.poses_gt[cls_id] = []
            self.scores_pred[cls_id] = []
            self.scores_gt[cls_id] = []
            self.ang_distances_cls[cls_id] = []
            self.ang_euler_distances_cls[cls_id] = [[],[],[]]
            self.ang_euler_distances_model_frame_cls[cls_id] = [[],[],[]]
            self.trans_distances_cls[cls_id] = []
            self.trans_distances_cls_non_abs[cls_id] = []
            self.z_distances_cls[cls_id] = []
            self.z_distances_cls_non_abs[cls_id] = []
            self.x_distances_cls_non_abs[cls_id] = []
            self.y_distances_cls_non_abs[cls_id] = []
            self.add_cls[cls_id] = []
            self.adds_cls[cls_id] = []
            self.adds_tp_cls[cls_id] = 0
            self.scores_cls[cls_id] = []

            self.ONEcmONEdeg_tp_cls[cls_id] = 0
            self.TWOcmTWOdeg_tp_cls[cls_id] = 0
            self.FIVEcmFIVEdeg_tp_cls[cls_id] = 0
            self.ONEcm_tp_cls[cls_id] = 0
            self.TWOcm_tp_cls[cls_id] = 0
            self.FIVEcm_tp_cls[cls_id] = 0
            self.ONEdeg_tp_cls[cls_id] = 0
            self.TWOdeg_tp_cls[cls_id] = 0
            self.FIVEdeg_tp_cls[cls_id] = 0

    def update(
        self,
        pred_trans,
        pred_rot,
        gt_trans,
        gt_rot,
        gt_cls,
        name_img,
        pred_cls=None,
        pred_bboxes=None,
        pred_scores=None,
        gt_scores=None,
        gt_bboxes=None,
        gt_groups=None,
        mapping=None,
    ):
        if self.matcher is not None:
            idx, gt_idx = self._match(
                pred_bboxes=pred_bboxes,
                pred_scores=pred_scores,
                gt_bboxes=gt_bboxes,
                gt_cls=gt_cls,
                gt_groups=gt_groups,
            )
            pred_trans, gt_trans = pred_trans[idx], gt_trans[gt_idx]
            pred_rot, gt_rot = pred_rot[idx], gt_rot[gt_idx]
            gt_cls = gt_cls[gt_idx]
        if mapping:
            gt_cls = torch.tensor([mapping[c.item()] for c in gt_cls])

        self.name_img += name_img

        pred_pose = (
            torch.cat([pred_rot, pred_trans.unsqueeze(-1)], dim=-1)
            .float()
            .cpu()
            .numpy()
        )
        gt_pose = (
            torch.cat([gt_rot, gt_trans.unsqueeze(-1)], dim=-1).float().cpu().numpy()
        )
        # if not 'testreal' in str(self.save_dir): 
        #     print('Careful, the convention of the rotation matrix is different for the synthetic dataset, we need to change it to the convention of the realdataset, which is x = roll, y = pitch z = yaw')
        
        matrice_rotation = np.array([[-1, 0, 0], [0, 0.0, 1], [0, 1, 0]])
        for i, pose in enumerate(gt_pose):
            rot = pose[:3, :3]
            trans = pose[:3, 3].reshape(3, 1)
            self.x_position.append(trans[0][0])
            self.y_position.append(trans[1][0])
            self.z_position.append(trans[2][0])
            #if not 'testreal' in str(self.save_dir): #No need for this condition has every prediction is passed to PoseMetric with the synthetic convention ! 
            rot = np.dot(rot, np.linalg.inv(matrice_rotation))
            cam_orig_m = -np.linalg.inv(rot).dot(trans)
            # Azimuth from [0, 360].
            azimuth = math.atan2(cam_orig_m[1, 0], cam_orig_m[0, 0])
            if azimuth < 0:
                azimuth += 2.0 * math.pi
            self.azimuth_angles.append((180.0 / math.pi) * azimuth)

            # Elevation from [-90, 90].
            a = np.linalg.norm(cam_orig_m)
            b = np.linalg.norm([cam_orig_m[0, 0], cam_orig_m[1, 0], 0])
            elev = math.acos(b / a)
            if cam_orig_m[2, 0] < 0:
                elev = -elev
            self.elevation_angles.append((180.0 / math.pi) * elev)

            gt_r = spatial.transform.Rotation.from_matrix(rot)

            gt_r = gt_r.as_euler(seq='xyz', degrees = True)#*180/np.pi
            self.roll_angles.append(gt_r[0]) # Roll from [-180, 180].
            self.pitch_angles.append(gt_r[1]) # Pitch from [-90, 90].
            self.yaw_angles.append(gt_r[2]) # Yaw from [-180, 180].

        gt_cls = gt_cls.long().cpu().numpy()

        eucl_distance = self._get_eucledian_distance(pred_pose, gt_pose)
        trans_distance_non_abs = calc_translation_error_batch_non_abs(pred_pose[:, :3, 3], gt_pose[:, :3, 3])
        z_distance = self._get_z_distance(pred_pose, gt_pose)
        z_distance_non_abs = self._get_z_distance_non_abs(pred_pose, gt_pose)
        y_distance_non_abs = self._get_y_distance_non_abs(pred_pose, gt_pose)
        x_distance_non_abs = self._get_x_distance_non_abs(pred_pose, gt_pose)
        ang_distance = self._get_angle_distance(pred_pose, gt_pose)
        ang_euler_distance = self._get_euler_angle_distance(pred_rot, gt_rot)
        ang_euler_distance_model_frame = self._get_euler_angle_distance_model_frame(pred_pose, gt_pose)
        if self.model_pts:
            add, adds = self._get_add_adds(gt_cls, pred_pose, gt_pose)

        score_l1 = None
        if self.pred_cls:
            pred_scores = pred_scores.cpu().numpy()
            gt_scores = gt_scores.cpu().numpy()
            score_l1 = np.abs(pred_scores - gt_scores).mean(axis=1)

        for i, obj in enumerate(gt_cls):
            obj = obj.item()
            self.num[obj] += 1
            self.poses_gt[obj].append(gt_pose[i])
            self.poses_pred[obj].append(pred_pose[i])
            self.trans_distances_cls[obj].append(eucl_distance[i])
            self.trans_distances_cls_non_abs[obj].append(trans_distance_non_abs[i])
            self.z_distances_cls[obj].append(z_distance[i])
            self.z_distances_cls_non_abs[obj].append(z_distance_non_abs[i])
            self.x_distances_cls_non_abs[obj].append(x_distance_non_abs[i])
            self.y_distances_cls_non_abs[obj].append(y_distance_non_abs[i])
            self.ang_distances_cls[obj].append(ang_distance[i])
            self.ang_euler_distances_cls[obj][0].append(ang_euler_distance[0][i])
            self.ang_euler_distances_cls[obj][1].append(ang_euler_distance[1][i])
            self.ang_euler_distances_cls[obj][2].append(ang_euler_distance[2][i])
            self.ang_euler_distances_model_frame_cls[obj][0].append(ang_euler_distance_model_frame[0][i])
            self.ang_euler_distances_model_frame_cls[obj][1].append(ang_euler_distance_model_frame[1][i])
            self.ang_euler_distances_model_frame_cls[obj][2].append(ang_euler_distance_model_frame[2][i])
            if self.model_pts:
                self.add_cls[obj].append(add[i])
                self.adds_cls[obj].append(adds[i])
                if adds[i] < self.diameter_thresholds[obj]:
                    self.adds_tp_cls[obj] += 1
            if score_l1 is not None:
                self.scores_cls[obj].append(score_l1[i])
                self.scores_pred[obj].append(pred_scores[i])
                self.scores_gt[obj].append(gt_scores[i])
                    
            if ang_distance[i] < 5 and eucl_distance[i] < 0.05:
                self.FIVEcmFIVEdeg_tp_cls[obj] += 1
                if ang_distance[i] < 2 and eucl_distance[i] < 0.02:
                    self.TWOcmTWOdeg_tp_cls[obj] += 1
                    if ang_distance[i] < 1 and eucl_distance[i] < 0.01:
                        self.ONEcmONEdeg_tp_cls[obj] += 1
            if eucl_distance[i] < 0.05:
                self.FIVEcm_tp_cls[obj] += 1
                if eucl_distance[i] < 0.02:
                    self.TWOcm_tp_cls[obj] += 1
                    if eucl_distance[i] < 0.01:
                        self.ONEcm_tp_cls[obj] += 1
            if ang_distance[i] < 5:
                self.FIVEdeg_tp_cls[obj] += 1
                if ang_distance[i] < 2:
                    self.TWOdeg_tp_cls[obj] += 1
                    if ang_distance[i] < 1:
                        self.ONEdeg_tp_cls[obj] += 1

    @property
    def num_targets(self) -> int:
        """Return the number of targets."""
        return sum(self.num.values())

    @property
    def avg_ang_distance(self) -> float:
        """Return the average angle distance."""
        if self.num_targets == 0:
            return float("inf")
        total_ang_distance = sum(
            sum((a for a in ang_list if a != float("inf")))
            for ang_list in self.ang_distances_cls.values()
        )
        return total_ang_distance / self.num_targets
    
    @property
    def avg_ang_euler_distance(self) -> list[float]:
        """Return the average angle distance."""
        if self.num_targets == 0:
            return float("inf")
        total_ang_distance = [
            [sum((a for a in ang_list[i] if a != float("inf")))  for i in range(3)]
            for ang_list in self.ang_euler_distances_cls.values()
        ]
        return [sum([obj_dist[i] for obj_dist in total_ang_distance]) / self.num_targets for i in range(3)]
    
    @property
    def avg_ang_euler_distance_model_frame(self) -> list[float]:
        """Return the average angle distance."""
        if self.num_targets == 0:
            return float("inf")
        total_ang_distance = [
            [sum((a for a in ang_list[i] if a != float("inf")))  for i in range(3)]
            for ang_list in self.ang_euler_distances_model_frame_cls.values()
        ]
        return [sum([obj_dist[i] for obj_dist in total_ang_distance]) / self.num_targets for i in range(3)]

    @property
    def avg_scores(self) -> float:
        """Return the average angle distance."""
        if self.num_targets == 0:
            return float("inf")
        total_scores = sum(
            sum((a for a in score_list if a != float("inf")))
            for score_list in self.scores_cls.values()
        )
        return total_scores / self.num_targets

    @property
    def avg_trans_distance(self) -> float:
        """Return the average translation distance."""
        if self.num_targets == 0:
            return float("inf")
        total_trans_distance = sum(
            sum((t for t in trans_list if t != float("inf")))
            for trans_list in self.trans_distances_cls.values()
        )
        return total_trans_distance / self.num_targets
    
    @property
    def avg_trans_distance_non_abs(self) -> float:
        """Return the average translation distance."""
        if self.num_targets == 0:
            return float("inf")
        total_trans_distance = np.sum(
            np.sum((t for t in trans_list if t != float("inf")))
            for trans_list in self.trans_distances_cls_non_abs.values()
        )
        return float(total_trans_distance / self.num_targets)

    @property
    def avg_z_distance(self) -> float:
        """Return the average translation distance."""
        if self.num_targets == 0:
            return float("inf")
        total_z_distance = sum(
            sum((t for t in z_list if t != float("inf")))
            for z_list in self.z_distances_cls.values()
        )
        return total_z_distance / self.num_targets
    
    @property
    def avg_z_distance_non_abs(self) -> float:
        """Return the average translation distance."""
        if self.num_targets == 0:
            return float("inf")
        total_z_distance = np.sum(
            np.sum((t for t in z_list if t != float("inf")))
            for z_list in self.z_distances_cls_non_abs.values()
        )
        return total_z_distance / self.num_targets
    
    @property
    def avg_x_distance_non_abs(self) -> float:
        """Return the average translation distance."""
        if self.num_targets == 0:
            return float("inf")
        total_x_distance = np.sum(
            np.sum((t for t in x_list if t != float("inf")))
            for x_list in self.x_distances_cls_non_abs.values()
        )
        return total_x_distance / self.num_targets
    
    @property
    def avg_y_distance_non_abs(self) -> float:
        """Return the average translation distance."""
        if self.num_targets == 0:
            return float("inf")
        total_y_distance = np.sum(
            np.sum((t for t in y_list if t != float("inf")))
            for y_list in self.y_distances_cls_non_abs.values()
        )
        return total_y_distance / self.num_targets

    @property
    def avg_adds(self) -> float:
        """Return the average ADD-S metric."""
        if self.num_targets == 0:
            return float("inf")
        total_add = sum(
            sum((a for a in add_list if a != float("inf")))
            for add_list in self.adds_cls.values()
        )
        return total_add / self.num_targets

    @property
    def avg_add(self) -> float:
        """Return the average ADD metric."""
        if self.num_targets == 0:
            return float("inf")
        total_add = sum(
            sum((a for a in add_list if a != float("inf")))
            for add_list in self.add_cls.values()
        )
        return total_add / self.num_targets 

    @property
    def avg_adds_acc(self) -> float:
        """Return the average ADD-S metric."""
        if self.num_targets == 0:
            return float("inf")
        # total_add = sum(self.adds_tp_cls.values())  # / len(self.adds_tp_cls)
        return (
            sum(self.avg_adds_acc_cls.values()) / len(self.avg_adds_acc_cls) 
        ) 
        # return total_add / self.num_targets * 100
    
    @property
    def avg_1cm1deg(self) -> float:
        """Return the average 1cm1° metric."""
        if self.num_targets == 0:
            return float("inf")
        # total_add = sum(self.adds_tp_cls.values())  # / len(self.adds_tp_cls)
        return (
            sum(self.avg_1cm1deg_cls.values()) / len(self.avg_1cm1deg_cls) 
        ) 
    


    @property
    def avg_trans_distance_cls(self) -> dict[int, float]:
        return {
            k: sum(v) / self.num[k] if self.num[k] != 0 else float("inf")
            for k, v in self.trans_distances_cls.items()
        }
    
    @property
    def avg_trans_distance_cls_non_abs(self) -> dict[int, float]:
        return {
            k: np.sum(v) / self.num[k] if self.num[k] != 0 else float("inf")
            for k, v in self.trans_distances_cls_non_abs.items()
        }

    @property
    def var_trans_distance_cls(self) -> dict[int, float]:
        return {
            k: np.var(v) if self.num[k] != 0 else float("inf")
            for k, v in self.trans_distances_cls.items()
        }
    
    @property
    def var_trans_distance_cls_non_abs(self) -> dict[int, float]:
        return {
            k: np.var(v) if self.num[k] != 0 else float("inf")
            for k, v in self.trans_distances_cls_non_abs.items()
        }

    @property
    def avg_z_distance_cls(self) -> dict[int, float]:
        return {
            k: sum(v) / self.num[k] if self.num[k] != 0 else float("inf")
            for k, v in self.z_distances_cls.items()
        }
    
    @property
    def avg_z_distance_cls_non_abs(self) -> dict[int, float]:
        return {
            k: np.sum(v) / self.num[k] if self.num[k] != 0 else float("inf")
            for k, v in self.z_distances_cls_non_abs.items()
        }
    
    @property
    def avg_x_distance_cls_non_abs(self) -> dict[int, float]:
        return {
            k: np.sum(v) / self.num[k] if self.num[k] != 0 else float("inf")
            for k, v in self.x_distances_cls_non_abs.items()
        }
    
    @property
    def avg_y_distance_cls_non_abs(self) -> dict[int, float]:
        return {
            k: np.sum(v) / self.num[k] if self.num[k] != 0 else float("inf")
            for k, v in self.y_distances_cls_non_abs.items()
        }
    
    @property
    def var_z_distance_cls(self) -> dict[int, float]:
        return {
            k: np.var(v) if self.num[k] != 0 else float("inf")
            for k, v in self.z_distances_cls.items()
        }
    
    @property
    def var_z_distance_cls_non_abs(self) -> dict[int, float]:
        return {
            k: np.var(v) if self.num[k] != 0 else float("inf")
            for k, v in self.z_distances_cls_non_abs.items()
        }
    
    @property
    def var_x_distance_cls_non_abs(self) -> dict[int, float]:
        return {
            k: np.var(v) if self.num[k] != 0 else float("inf")
            for k, v in self.x_distances_cls_non_abs.items()
        }
    
    @property
    def var_y_distance_cls_non_abs(self) -> dict[int, float]:
        return {
            k: np.var(v) if self.num[k] != 0 else float("inf")
            for k, v in self.y_distances_cls_non_abs.items()
        }
    
    @property
    def avg_ang_distance_cls(self) -> dict[int, float]:
        return {
            k: sum(v) / self.num[k] if self.num[k] != 0 else float("inf")
            for k, v in self.ang_distances_cls.items()
        }
    
    @property
    def avg_ang_euler_distance_cls(self) -> dict[int, list[float]]:
        return {
            k: [sum(v) / self.num[k] if self.num[k] != 0 else float("inf") for v in self.ang_euler_distances_cls[k]]
            for k in self.ang_euler_distances_cls.keys()
        }
    
    @property
    def avg_ang_euler_distance_model_frame_cls(self) -> dict[int, list[float]]:
        return {
            k: [sum(v) / self.num[k] if self.num[k] != 0 else float("inf") for v in self.ang_euler_distances_model_frame_cls[k]]
            for k in self.ang_euler_distances_model_frame_cls.keys()
        }

    @property
    def var_ang_distance_cls(self) -> dict[int, float]:
        return {
            k: np.var(v) if self.num[k] != 0 else float("inf")
            for k, v in self.ang_distances_cls.items()
        }
    
    @property
    def var_ang_euler_distance_cls(self) -> dict[int, list[float]]:
        return {
            k: [np.var(v) if self.num[k] != 0 else float("inf") for v in self.ang_euler_distances_cls[k]]
            for k in self.ang_euler_distances_cls.keys()
        }

    @property
    def var_ang_euler_distance_model_frame_cls(self) -> dict[int, list[float]]:
        return {
            k: [np.var(v) if self.num[k] != 0 else float("inf") for v in self.ang_euler_distances_model_frame_cls[k]]
            for k in self.ang_euler_distances_model_frame_cls.keys()
        }

    @property
    def avg_adds_cls(self) -> dict[int, float]:
        return {
            k: sum(v) / self.num[k] if self.num[k] != 0 else float("inf")
            for k, v in self.adds_cls.items()
        }

    @property
    def avg_add_cls(self) -> dict[int, float]:
        return {
            k: sum(v) / self.num[k] if self.num[k] != 0 else float("inf")
            for k, v in self.add_cls.items()
        }

    @property
    def avg_adds_acc_cls(self) -> dict[int, float]:
        return {
            k: v / self.num[k] * 100 if self.num[k] != 0 else float("inf")
            for k, v in self.adds_tp_cls.items()
        } 

    @property
    def avg_scores_cls(self) -> dict[int, float]:
        return {
            k: sum(v) / self.num[k] if self.num[k] != 0 else float("inf")
            for k, v in self.scores_cls.items()
        }
    
    @property
    def avg_1cm1deg_cls(self) -> dict[int, float]:
        return {
            k: v / self.num[k] * 100 if self.num[k] != 0 else float("inf")
            for k, v in self.ONEcmONEdeg_tp_cls.items()
        } 
    
    @property
    def tp_1cm1deg_cls(self) -> dict[int, float]:
        return {
            k: v if self.num[k] != 0 else float("inf")
            for k, v in self.ONEcmONEdeg_tp_cls.items()
        }

    @property
    def tp_2cm2deg_cls(self) -> dict[int, float]:
        return {
            k: v  if self.num[k] != 0 else float("inf")
            for k, v in self.TWOcmTWOdeg_tp_cls.items()
        }

    @property
    def tp_5cm5deg_cls(self) -> dict[int, float]:
        return {
            k: v  if self.num[k] != 0 else float("inf")
            for k, v in self.FIVEcmFIVEdeg_tp_cls.items()
        }

    @property
    def tp_1cm_cls(self) -> dict[int, float]:
        return {
            k: v  if self.num[k] != 0 else float("inf")
            for k, v in self.ONEcm_tp_cls.items()
        }

    
    @property
    def tp_2cm_cls(self) -> dict[int, float]:
        return {
            k: v  if self.num[k] != 0 else float("inf")
            for k, v in self.TWOcm_tp_cls.items()
        }

    
    @property
    def tp_5cm_cls(self) -> dict[int, float]:
        return {
            k: v  if self.num[k] != 0 else float("inf")
            for k, v in self.FIVEcm_tp_cls.items()
        }

    @property
    def tp_1deg_cls(self) -> dict[int, float]:
        return {
            k: v  if self.num[k] != 0 else float("inf")
            for k, v in self.ONEdeg_tp_cls.items()
        }

    @property
    def tp_2deg_cls(self) -> dict[int, float]:
        return {
            k: v  if self.num[k] != 0 else float("inf")
            for k, v in self.TWOdeg_tp_cls.items()
        }

    
    @property
    def tp_5deg_cls(self) -> dict[int, float]:
        return {
            k: v  if self.num[k] != 0 else float("inf")
            for k, v in self.FIVEdeg_tp_cls.items()
        }
    
    
    @property
    def tp_thresholds_cls(self) -> dict[int, dict[str, float]]:
        return {k: {
            "1cm1deg": self.tp_1cm1deg_cls[k],
            "2cm2deg": self.tp_2cm2deg_cls[k],
            "5cm5deg": self.tp_5cm5deg_cls[k],
            "1cm": self.tp_1cm_cls[k],
            "2cm": self.tp_2cm_cls[k],
            "5cm": self.tp_5cm_cls[k],
            "1deg": self.tp_1deg_cls[k],
            "2deg": self.tp_2deg_cls[k],
            "5deg": self.tp_5deg_cls[k],
        } for k in self.class_names.keys()}

    @property
    def mean_results(self) -> list[float]:
        """Return the metrics."""
        return [
            self.avg_adds_acc,
            self.avg_adds,
            self.avg_trans_distance,
            self.avg_z_distance,
            self.avg_ang_distance,
            self.avg_scores,
            self.avg_1cm1deg,
        ]

    @property
    def avg_metrics_cls(self) -> dict[int, list[float]]:
        avg_metrics_cls = {}
        for k, ang_cls in self.avg_ang_distance_cls.items():
            avg_metrics_cls[k] = [
                self.avg_adds_acc_cls[k],
                self.avg_adds_cls[k],
                self.avg_trans_distance_cls[k],
                self.avg_z_distance_cls[k],
                ang_cls,
                self.avg_scores_cls[k],
                self.avg_1cm1deg_cls[k],
            ]
        return avg_metrics_cls

    @property
    def fitness(self) -> float:
        """Return the fitness of the metrics."""
        w = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return self.mean_results.copy()[
            0
        ]  

    @property
    def keys(self) -> list[str]:
        return [
            "ADD(-S)",
            "ADD(-S)[m]",
            "eucl. dist.",
            "z dist.",
            "ang. dist.",
            "score l1",
            "1cm1°",
        ]

    @property
    def results_dict(self):
        """Returns a dict of all metrics."""
        return dict(
            zip(
                self.keys + ["fitness"],
                self.mean_results + [self.fitness],
            )
        )

    def _get_add_adds(self, gt_cls, pred_pose, gt_pose):
        add = []
        adds = []
        for obj_cls, pose, gt_pose in zip(gt_cls, pred_pose, gt_pose):
            model_pts = self.model_pts[obj_cls]
            error = calc_add(model_pts, pose, gt_pose).astype(np.float32)
            add.append(error)
            if self.model_symmetries[obj_cls]:
                error = calc_adi(model_pts, pose, gt_pose).astype(np.float32)
            adds.append(error)
        return add, adds

    def _get_eucledian_distance(self, pose, gt_pose):
        return calc_translation_error_batch(pose[:, :3, 3], gt_pose[:, :3, 3])

    def _get_z_distance(self, pose, gt_pose):
        return np.abs(pose[:, -1, -1] - gt_pose[:, -1, -1])
    
    def _get_z_distance_non_abs(self, pose, gt_pose):
        return pose[:, -1, -1] - gt_pose[:, -1, -1]
    
    def _get_x_distance_non_abs(self, pose, gt_pose):
        return pose[:, 0, -1] - gt_pose[:, 0, -1]
    
    def _get_y_distance_non_abs(self, pose, gt_pose):
        return pose[:, 1, -1] - gt_pose[:, 1, -1]

    def _get_angle_distance(self, pose, gt_pose):
        return calc_rotation_error_batch(pose[:, :3, :3], gt_pose[:, :3, :3])
    
    def _get_euler_angle_distance(self, pose, gt_pose):
        return calc_euler_angle_error_batch(pose[:, :3, :3], gt_pose[:, :3, :3], seq = 'zyx')
    
    def _get_euler_angle_distance_model_frame(self, pose, gt_pose):
        return calc_euler_angle_error_batch_model_frame(pose[:, :3, :3], gt_pose[:, :3, :3], seq = 'zyx')

    def _match(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups):
        """
        Match the predicted bboxes with the ground truth bboxes
        """
        match_indices = self.matcher(
            pred_bboxes=pred_bboxes,
            pred_scores=pred_scores,
            gt_bboxes=gt_bboxes,
            gt_cls=gt_cls,
            gt_groups=gt_groups,
        )
        idx, gt_idx = self._get_index(match_indices)
        return idx, gt_idx

    def _get_index(self, match_indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)]
        )
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    def save(self):
        metrics = {
            "num": self.num,
            "name_img": self.name_img,

            "azimuth_angles": self.azimuth_angles,
            "elevation_angles": self.elevation_angles,
            "roll_angles": self.roll_angles,
            "pitch_angles": self.pitch_angles,
            "yaw_angles": self.yaw_angles,
            "x_position": self.x_position,
            "y_position": self.y_position,
            "z_position": self.z_position,

            "poses_pred": self.poses_pred,
            "poses_gt": self.poses_gt,
            "ang_distances_cls": self.ang_distances_cls,
            "ang_euler_distances_cls": self.ang_euler_distances_cls,
            "ang_euler_distances_model_frame_cls": self.ang_euler_distances_model_frame_cls,
            "trans_distances_cls": self.trans_distances_cls,
            "trans_distances_cls_non_abs": self.trans_distances_cls_non_abs,
            "z_distances_cls": self.z_distances_cls,
            "z_distances_cls_non_abs": self.z_distances_cls_non_abs,
            "x_distances_cls_non_abs": self.x_distances_cls_non_abs,
            "y_distances_cls_non_abs": self.y_distances_cls_non_abs,
            "add_cls": self.add_cls,
            "adds_cls": self.adds_cls,
            "tp_thresholds_cls": self.tp_thresholds_cls,
        }
        with open(str(self.save_dir / "metrics.pkl"), "wb+") as f:
            pkl.dump(metrics, f)

        # with open(str(self.save_dir / "metrics.pkl"), "wb+") as f:
        #     pkl.dump(self, f)

    # class Evaluator:
    # def __init__(self, path):
    # self.path = path
    # self._unpack()

    # def _unpack(self):
    # with open(self.path, "rb") as f:
    # metrics = pkl.load(f)
    # self.num = metrics["num"]
    # self.poses_pred = metrics["poses_pred"]
    # self.poses_gt = metrics["poses_gt"]
    # self.ang_distances_cls = metrics["ang_distances_cls"]
    # self.trans_distancs_cls = metrics["trans_distances_cls"]
    # self.add_cls = metrics["add_cls"]
    # self.adds_cls = metrics["adds_cls"]

    def evaluate_all(self):
        with open(self.save_dir / "eval.log", "w+") as log_file:
            # ADD
            # a = self.evaluate_auc_acc(errors_cls=self.add_cls)
            self.write_log_file(
                log_file, *self.evaluate_auc_acc(errors_cls=self.add_cls), "ADD"
            )
            # ADD-S
            self.write_log_file(
                log_file, *self.evaluate_auc_acc(errors_cls=self.adds_cls), "ADD(-S)"
            )
            # # TRANS
            # self.write_trans_log_file(log_file)

            # # ANG
            # self.write_ang_log_file(log_file)

            # # TRANS+ANG
            # self.write_trans_ang_log_file(log_file)

    def write_trans_log_file(self, log_file):
        log_file.write(
            "\n* {} *\n {:^}\n* {} *\n".format(
                "-" * 100, "Metric Average Translation Error in Centimeters", "-" * 100
            )
        )
        thresholds = [0.01, 0.02, 0.05]
        for threshold in thresholds:
            acc_cls = {}
            tp_acc = {k: 0 for k in self.class_names.keys()}
            for obj_cls, obj_name in self.class_names.items():
                for trans in self.trans_distances_cls[obj_cls]:
                    if trans < threshold:
                        tp_acc[obj_cls] += 1

                if self.num[obj_cls] > 0:
                    acc_cls[obj_cls] = tp_acc[obj_cls] / self.num[obj_cls] * 100
                    log_file.write(
                        f"{obj_name:22} \t\t Avg Error: {self.avg_trans_distance_cls[obj_cls]*100:.2f}cm  (std: {np.sqrt(self.var_trans_distance_cls[obj_cls])*100:.2f}cm)  \t\t Accuracy({threshold*100}cm): {acc_cls[obj_cls]:.2f} ({tp_acc[obj_cls]}/{self.num[obj_cls]}) \n"
                    )                   
            acc = sum(acc_cls.values()) / len(acc_cls)
            log_file.write(
                f"Average: {self.avg_trans_distance * 100:.2f}, Overall Accuracy: {acc:.2f} with threshold: {threshold}m \n \n"
            )

    def write_ang_log_file(self, log_file):
        log_file.write(
            "\n* {} *\n {:^}\n* {} *\n".format(
                "-" * 100, "Metric Average Rotation Error in Degrees", "-" #* 100
            )
        )

        thresholds = [1, 2, 5]
        for threshold in thresholds:
            acc_cls = {}
            tp_acc = {k: 0 for k in self.class_names.keys()}
            for obj_cls, obj_name in self.class_names.items():
                for trans in self.ang_distances_cls[obj_cls]:
                    if trans < threshold:
                        tp_acc[obj_cls] += 1

                if self.num[obj_cls] > 0:
                    acc_cls[obj_cls] = tp_acc[obj_cls] / self.num[obj_cls] * 100
                    log_file.write(
                        f"{obj_name:22} \t\t Avg Error: {self.avg_ang_distance_cls[obj_cls]:.2f} \t\t Accuracy({threshold} degree): {acc_cls[obj_cls]:.2f} \n"
                    )

            acc = sum(acc_cls.values()) / len(acc_cls)
            log_file.write(
                f"Average: {self.avg_ang_distance:.2f}, Overall Accuracy: {acc} with threshold: {threshold} degree,  \n \n"
            )

    def write_trans_ang_log_file(self, log_file):
        log_file.write(
            "\n* {} *\n {:^}\n* {} *\n".format(
                "-" * 100, "Metric Average Translation Error in Centimeters and Angle Error", "-" * 100
            )
        )
        thresholds = [0.01, 0.02, 0.05]
        for threshold in thresholds:
            acc_cls = {}
            tp_acc = {k: 0 for k in self.class_names.keys()}
            for obj_cls, obj_name in self.class_names.items():
                for i in range(len(self.trans_distances_cls[obj_cls])):
                    if self.trans_distances_cls[obj_cls][i] < threshold and self.ang_distances_cls[obj_cls][i] < threshold*100:
                        tp_acc[obj_cls] += 1

                if self.num[obj_cls] > 0:
                    acc_cls[obj_cls] = tp_acc[obj_cls] / self.num[obj_cls] * 100
                    log_file.write(
                        f"{obj_name:22} \t\t Accuracy({int(threshold*100)}cm{int(threshold*100)}°): {acc_cls[obj_cls]:.2f} \n"
                    )
                    
            acc = sum(acc_cls.values()) / len(acc_cls)
            log_file.write(
                f"Average: {self.avg_trans_distance * 100:.2f}, Overall Accuracy: {acc:.2f} with threshold: {int(threshold*100)}cm{int(threshold*100)}°\n \n"
            )

    def evaluate_auc_acc(self, errors_cls):
        dx = 0.01
        auc_thresholds = [i * 0.1 * dx for i in range(int(1 / dx))]
        # diameter_thresholds = {
        #    k: 0.1 * diameter for k, diameter in self.diameters.items()
        # }

        acc_mean_cls = {}
        acc_diameter_cls = {}

        tp_diameter = {k: 0 for k in self.class_names.keys()}
        error_cls = copy.deepcopy(errors_cls)
        for obj_cls in self.class_names.keys():
            tp_auc = {threshold: 0 for threshold in auc_thresholds}
            for error in error_cls[obj_cls]:
                if error < self.diameter_thresholds[obj_cls]:
                    tp_diameter[obj_cls] += 1
                for threshold in auc_thresholds:
                    if error < threshold:
                        tp_auc[threshold] += 1

            if self.num[obj_cls] > 0:
                acc_diameter_cls[obj_cls] = (
                    tp_diameter[obj_cls]
                    / self.num[obj_cls]
                    * 100
                    #    if self.num[obj_cls] > 0
                    #    else 0
                )
                acc_mean_cls[obj_cls] = (
                    simps(np.array(list(tp_auc.values())) / self.num[obj_cls], dx=dx)
                    # / 0.1
                    * 100
                )  # if self.num[obj_cls] > 0 else 0

        # TODO: in length also the zero values get included
        acc_mean = sum(acc_mean_cls.values()) / len(acc_mean_cls)
        acc_diameter = sum(acc_diameter_cls.values()) / len(acc_diameter_cls)
        return (acc_mean_cls, acc_mean, acc_diameter_cls, acc_diameter, tp_diameter)

    def write_log_file(
        self,
        log_file,
        acc_mean_cls,
        acc_mean,
        acc_diameter_cls,
        acc_diameter,
        tp_diameter,
        name="ADD(-S)",
    ):
        """Writes the accuracy and error to the log file."""

        log_file.write("\n* {} *\n {:^}\n* {} *\n".format("-" * 100, name, "-" * 100))

        # for obj_cls, obj_name in self.class_names.items():
        for obj_cls in acc_mean_cls.keys():
            obj_name = self.class_names[obj_cls]
            # log_file.write(f"\n** {obj_name} **\n")
            avg_error = (
                self.avg_adds_cls[obj_cls] * 100
                if name == "ADD(-S)"
                else self.avg_add_cls[obj_cls] * 100
            )
            log_file.write(
                f"{obj_name:22} \t\t Avg Error: {avg_error:.2f} \t\t AUC: {acc_mean_cls[obj_cls]:.2f} \t\t {name}: {acc_diameter_cls[obj_cls]:.2f} ({tp_diameter[obj_cls]}/{self.num[obj_cls]})\n"
            )

            # log_file.write(f"average error in cm: {avg_error:.2f} \n")
            # log_file.write(
            #    f"threshold=[0.0, 0.10], area: {acc_mean_cls[obj_cls]:.2f} \n"
            # )
            # log_file.write(
            #    "threshold=0.1d, correct poses: {}, all poses: {}, accuracy: {:.2f}\n".format(
            #        tp_diameter[obj_cls], self.num[obj_cls], acc_diameter_cls[obj_cls]
            #    )
            # )
        log_file.write("=" * 30 + "\n")
        log_file.write(
            f"---------- {name} performance over {len(acc_mean_cls)}/{len(self.class_names)} classes ------------\n"
        )
        log_file.write(
            "average error in cm: {:.2f} \n".format(
                self.avg_adds * 100 if name == "ADD(-S)" else self.avg_add * 100
            )
        )
        log_file.write("threshold=[0.0, 0.10], area: {:.2f}".format(acc_mean))
        log_file.write("\n")
        log_file.write(
            "threshold=0.1d, mean accuracy: {:.2f}\n \n".format(acc_diameter)
        )