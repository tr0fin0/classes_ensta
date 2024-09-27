"""
Reading a laser scan dataset
author: David Filliat
"""

import os

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import icp

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_scan(ranges, angles, pose):
    """
    Create a scan dict from its components.
    Args:
        ranges: np.array of scan distances
        angles: np.array of angles of each reading
        pose: absolute pose as np.array [x,y,theta]
    Returns:
        scan: a dict with ranges, angles, pose, x & y coordinates of scan points
    """

    scan = {'ranges': np.array(ranges),
            'angles': np.array(angles),
            'pose': pose.reshape(-1),
            'x': pose[0] + np.multiply(ranges, np.cos(angles + pose[2])),
            'y': pose[1] + np.multiply(ranges, np.sin(angles + pose[2])),
            }

    return scan


def transform_scan(scan, R, t):
    """
    Change the pose of a scan with rotation matrix and translation vector.
    Args:
        scan: scan structure from 'create_scan' function
        R: 2x2 rotation matrix
        t: 2x1 translation vector
    Returns:
        newscan: transformed scan
    """
    pose = scan["pose"]
    newXY = np.matmul(R, pose[0:2].reshape(2, -1)) + t
    newTheta = pose[2] + math.atan2(R[1, 0], R[0, 0])

    new_scan = {
        "pose": np.array([newXY[0, 0], newXY[1, 0], newTheta]).reshape(-1),
        "ranges": scan["ranges"],
        "angles": scan["angles"],
        "x": newXY[0, 0] + np.multiply(scan["ranges"], np.cos(scan["angles"] + newTheta)),
        "y": newXY[1, 0] + np.multiply(scan["ranges"], np.sin(scan["angles"] + newTheta)),
    }
    return new_scan


def update_scan_pose(scan, new_pose):
    """
    Update the pose of a scan to a new pose.
    Args:
        scan: scan structure from 'create_scan' function
        newPose: new pose as np.array [x,y,theta]
    Returns:
        newscan: updated scan
    """

    new_scan = {
        "pose": new_pose.reshape(-1),
        "ranges": scan["ranges"],
        "angles": scan["angles"],
        "x": new_pose[0] + np.multiply(scan["ranges"], np.cos(scan["angles"] + new_pose[2])),
        "y": new_pose[1] + np.multiply(scan["ranges"], np.sin(scan["angles"] + new_pose[2])),
    }
    return new_scan
    

def find_closest_scan(map, scan):
    """
    Return map scan ids sorted according to distance to scan.
    Args:
        map: list of previous scans
        scan: current scan
    Returns:
        distances: sorted distances
        sorted_id: sorted indices
    """

    def distance(scan1, scan2):
        """
        Computes distance between two scan poses.
        Args:
            scan1: first scan
            scan2: second scan
        Returns:
            dist: distance between scans
        """
        dist = np.linalg.norm(scan1["pose"][0:2] - scan2["pose"][0:2]) + abs(icp.angle_wrap(scan1["pose"][2] - scan2["pose"][2]))/15
        # The following is to prevent matching scans with too large rotation
        # differences
        if abs(icp.angle_wrap(scan1["pose"][2]-scan2["pose"][2])) > np.pi/3:
            dist = dist * 2          
        return dist

    distances = np.array([distance(scan, previous_scan) for previous_scan in map])
    sorted_id = np.argsort(distances)
    return distances[sorted_id], sorted_id


def read_u2is_odom_entry(file):
    """
    Read the next odometry entry from the 'u2is' dataset.
    Args:
        file: file object
    Returns:
        odomTime: list of seconds and nanoseconds
        odomData: list of odometry data
    """

    secs = file.readline()
    secs = int(secs[10:])
    nsecs = file.readline()
    nsecs = int(nsecs[11:])
    odomTime = [secs, nsecs]
    odomData = []
    for i in range(13):
        odomData.append(float(file.readline()))

    return odomTime, odomData


def read_u2is_laser_entry(file):
    """
    Read the next laser scan entry from the 'u2is' dataset.
    Args:
        file: file object
    Returns:
        laserTime: list of seconds and nanoseconds
        laserData: np.array of laser data
    """

    secs = file.readline()
    secs = int(secs[10:])
    nsecs = file.readline()
    nsecs = int(nsecs[11:])
    laserTime = [secs, nsecs]
    line = file.readline()
    laserData = line[9:-2].split(',')
    laserData = np.array([float(i) for i in laserData])

    # print("Reading laser : " + str(laserTime))

    return laserTime, laserData


def read_u2is(num_scans):
    """
    Reading and formatting 'u2is' dataset.
    Args:
        number: number of scans to read
    Returns:
        scanList: list of dict with scans
    """

    if num_scans == 0 or num_scans > 855:
        num_scans = 845

    logging.info('Reading u2is dataset')


    print(os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset')))



    # U2IS_LASER_FILE = 'dataset/U2IS/laser_filt.txt'
    # U2IS_ODOM_FILE = 'dataset/U2IS/odom_filt.txt'



    U2IS_LASER_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset/U2IS/laser_filt.txt'))
    U2IS_ODOM_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset/U2IS/odom_filt.txt'))

    with open(U2IS_LASER_FILE, 'r') as laser_file, open(U2IS_ODOM_FILE, 'r') as odom_file:
        scanList = []
        odomTime, odomData = read_u2is_odom_entry(odom_file)
        angles = np.arange(-2.35619449615, 2.35619449615, 0.00436332309619)

        for _ in range(num_scans):
            laserTime, laserData = read_u2is_laser_entry(laser_file)
            while odomTime[0] > laserTime[0]:
                laserTime, laserData = read_u2is_laser_entry(laser_file)
            if odomTime[0] == laserTime[0]:
                while odomTime[1] > laserTime[1]:
                    laserTime, laserData = read_u2is_laser_entry(laser_file)

            laserData[laserData > 20.0] = np.inf
            laserData[:80] = np.inf
            laserData[-80:] = np.inf

            siny_cosp = 2.0 * (odomData[6] * odomData[5] + odomData[3] * odomData[4])
            cosy_cosp = 1.0 - 2.0 * (odomData[4] * odomData[4] + odomData[5] * odomData[5])
            yaw = math.atan2(siny_cosp, cosy_cosp)

            pose = np.array([odomData[0] + 0.1 * np.cos(yaw), odomData[1] + 0.1 * np.sin(yaw), yaw])
            scanList.append(create_scan(laserData[0::2], angles[0::2], pose))
            odomTime, odomData = read_u2is_odom_entry(odom_file)

    logging.info(f'Finished reading {len(scanList)} scans')
    return scanList


def read_fr079_odom_entry(file):
    """
    Read the next odometry entry from the 'fr079' dataset.
    Args:
        file: file object
    Returns:
        odomTime: list of seconds and nanoseconds
        odomData: list of odometry data
    """

    secs = file.readline()
    secs = int(secs[10:])
    nsecs = file.readline()
    nsecs = int(nsecs[11:])
    odomTime = [secs, nsecs]
    odomData = []
    for i in range(13):
        odomData.append(float(file.readline()))

    return odomTime, odomData


def read_fr079_laser_entry(file):
    """
    Read the next laser scan entry from the 'fr079' dataset.
    Args:
        file: file object
    Returns:
        laserTime: list of seconds and nanoseconds
        laserData: np.array of laser data
    """

    secs = file.readline()
    secs = int(secs[10:])
    nsecs = file.readline()
    nsecs = int(nsecs[11:])
    laserTime = [secs, nsecs]
    line = file.readline()
    laserData = line[9:-2].split(',')
    laserData = np.array([float(i) for i in laserData])

    # print("Reading laser : " + str(laserTime))

    return laserTime, laserData


def read_fr079(num_scans):
    """
    Reading and formatting 'fr079' dataset.
    Args:
        number: number of scans to read
    Returns:
        scanList: list of dict with scans
    """

    if num_scans == 0 or num_scans > 4919:
        num_scans = 4919

    logging.info('Reading FR079 dataset')

    laser_file = open('dataset/fr079/laserData.txt', 'r')

    scan_list = []

    # discard first line
    line_data = laser_file.readline()

    scan_angles = np.arange(-math.pi / 2, math.pi / 2, math.pi / 359)

    for i in range(num_scans):

        # Reading raw data
        line_data = laser_file.readline()
        raw_data = line_data[11:-20].split(' ')
        raw_data = np.array([float(i) for i in raw_data])

        # Create scanlist
        scan_list.append(create_scan(raw_data[0:360], scan_angles, raw_data[360:363]))

    laser_file.close()

    logging.info(f'Finished reading {len(scan_list)} scans')
    return scan_list

if __name__ == '__main__':
    read_u2is(10)
