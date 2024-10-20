"""
 Incremental ICP SLAM  - Basic implementation for teaching purpose only...
 Computes position of each scan with respect to closest one in the
 current map and add scan to the map if it is far enough from all the
 existing ones
 author: David Filliat
"""

import readDatasets as datasets
import matplotlib.pyplot as plt
import icp
import numpy as np
import copy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global parameters
AXIS_LIMIT = [-5.5, 12.5, -12.5, 6.5] # Axis limits for the u2is dataset
#AXIS_LIMIT = [-20, 25, -10, 30] # Axis limits for the fr079 dataset
DISPLAY = True # Display is very slow, set to False for batch processing

# Parameters for map building
DIST_THRESHOLD_ADD = 0.3
MIN_SCAN = 0
STEP = 3

def initialize_display(scan):
    """
    Initialize the display for the scans.
    Args:
        scan: Initial scan
    Returns:
        Tuple[plt.Figure, plt.Axes, plt.Axes]: The figure and axes for the plots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 7))
    color = np.random.rand(3,)
    ax1.scatter(scan["x"], scan["y"], color=color, s=1)
    ax1.scatter(scan["pose"][0], scan["pose"][1], color=color, s=3)
    ax1.axis(AXIS_LIMIT)
    ax1.set_title('Pose from raw odometry')
    ax2.scatter(scan["x"], scan["y"], color=color, s=1)
    ax2.scatter(scan["pose"][0], scan["pose"][1], color=color, s=3)
    ax2.axis(AXIS_LIMIT)
    ax2.set_title('Map after incremental ICP SLAM')
    if DISPLAY:
        plt.pause(0.01)
    return fig, ax1, ax2

def update_display(ax, scan):
    """
    Update the display with the new scan data.
    Args:
        ax: Axis to update
        scan: Scan to add to the display
    """
    color = np.random.rand(3,)
    ax.scatter(scan["x"], scan["y"], color=color, s=1)
    ax.scatter(scan["pose"][0], scan["pose"][1], color=color, s=3)
    if DISPLAY:
        plt.pause(0.01)

def incremental_SLAM():
    # Reading data
    # scan_list = datasets.read_fr079(0)
    scan_list = datasets.read_u2is(0)
    MAX_SCAN = len(scan_list) - STEP

    # Copy for reference display and map init
    odom_scan_list = copy.deepcopy(scan_list)
    map_scans = [scan_list[MIN_SCAN]]

    # Init displays
    fig, ax1, ax2 = initialize_display(odom_scan_list[MIN_SCAN])

    # Perform incremental SLAM
    for i in range(MIN_SCAN + STEP, MAX_SCAN, STEP):
        # Get list of map scans sorted by distance
        sorted_dist, sorted_id = datasets.find_closest_scan(map_scans, scan_list[i])
        ref_scan_id = sorted_id[0]
        logging.info(f'Matching new scan to reference scan {ref_scan_id}')

        # Perform ICP with closest scan
        R, t, error, iter = icp.icp(map_scans[ref_scan_id], scan_list[i], 200, 1e-7)

        # Correct all future scans' odometry pose
        for j in range(i, MAX_SCAN, STEP):
            scan_list[j] = datasets.transform_scan(scan_list[j], R, t)

        # Add scan to map if it is far enough
        if np.linalg.norm(scan_list[i]["pose"][0:2] - map_scans[ref_scan_id]["pose"][0:2]) > DIST_THRESHOLD_ADD:
            map_scans.append(scan_list[i])
            logging.info(f'Added to map, new size: {len(map_scans)}')
            update_display(ax2, map_scans[-1])

        # Update display
        update_display(ax1, scan_list[i])

    plt.savefig('icpIncrementalSLAM_Map.png')
    logging.info("Press Q in figure to finish...")
    plt.show()

if __name__ == "__main__":
    incremental_SLAM()