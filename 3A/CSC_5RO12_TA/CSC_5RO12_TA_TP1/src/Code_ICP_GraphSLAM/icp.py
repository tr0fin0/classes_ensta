"""
 Simple 2D ICP implementation
 author: David Filliat
"""

import numpy as np
from scipy.spatial import KDTree
import math

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# A few helper function

def angle_wrap(angle):
    """
    Keep angle between -pi and pi
    Args:
        a: angle in radian
    Returns:
        angle between -pi and pi
    """
    return np.fmod(angle + np.pi, 2*np.pi ) - np.pi


def mean_angle(angle_list):
    """
    Compute the mean of a list of angles
    Args:
        angleList: list of angles in radian
    Returns:
        mean angle in radian
    """

    mean_cos = np.mean(np.cos(angle_list))
    mean_sine = np.mean(np.sin(angle_list))

    return math.atan2(mean_sine, mean_cos)


def icp(reference_scan, scan_to_align, max_iterations, thres):
    """
    ICP (iterative closest point) algorithm
    Simple ICP implementation for teaching purpose
    Args:
        reference_scan : scan taken as the reference position
        scan_to_align : scan to align on the model
        maxIter : maximum number of ICP iterations
        thres : threshold to stop ICP when correction is smaller
    Returns:
        R : rotation matrix
        t : translation vector
        meandist : mean point distance after convergence
    """

    # Various inits
    olddist = float("inf")  # residual error
    maxRange = 10  # limit on the distance of points used for ICP

    # Create array of x and y coordinates of valid readings for reference scan
    valid = reference_scan["ranges"] < maxRange
    ref_points = np.array([reference_scan["x"], reference_scan["y"]])
    ref_points = ref_points[:, valid]

    # Create array of x and y coordinates of valid readings for processed scan
    valid = scan_to_align["ranges"] < maxRange
    scan_points = np.array([scan_to_align["x"], scan_to_align["y"]])
    scan_points = scan_points[:, valid]

    # ----------------------- TODO ------------------------
    # Filter data points too close to each other
    # Put the result in filtered_scan_points
    filtered_scan_points = scan_points

    # Initialize transformation to identity
    R = np.eye(2)
    t = np.zeros((2, 1))

    # Main ICP loop
    for iter in range(max_iterations):

        # ----- Find nearest Neighbors for each point, using kd-trees for speed
        tree = KDTree(ref_points.T)
        distance, index = tree.query(filtered_scan_points.T) # index gives the closest reference point for each scan point 
        meandist = np.mean(distance)

        # ----------------------- TODO ------------------------
        # filter point matchings, keeping only the closest ones
        # you have to modify :
        # - 'matched_scan_points' with the points that are kept
        # - 'index' with the entries of the points that are kept
        matched_scan_points = filtered_scan_points
        index = index


        # ----- Compute transform

        # Compute point mean
        mdat = np.mean(matched_scan_points, 1)
        mref = np.mean(ref_points[:, index], 1)

        # Use SVD for transform computation
        C = np.transpose(matched_scan_points.T-mdat) @ (ref_points[:, index].T - mref)
        u, s, vh = np.linalg.svd(C)
        Ri = vh.T @ u.T
        Ti = mref - Ri @ mdat

        # Apply transformation to points
        filtered_scan_points = Ri @ filtered_scan_points
        filtered_scan_points = np.transpose(filtered_scan_points.T + Ti)

        # Update global transformation
        R = Ri @ R
        t = Ri @ t + Ti.reshape(2, 1)

        # Stop when no more progress
        if abs(olddist-meandist) < thres:
            break

        # store mean residual error to check progress
        olddist = meandist

    logging.info("ICP with {:.3e} point error in {:d} iter".format(meandist, iter))

    return R, t, meandist, iter
