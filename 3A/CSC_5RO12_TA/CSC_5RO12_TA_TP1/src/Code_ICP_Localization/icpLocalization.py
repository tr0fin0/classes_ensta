"""
 Simple ICP localisation demo
 Compute position of each scan using ICP
 with respect to the previous one
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

def initialize_display(scan_list, min_scan):
    """
    Initialize the display for the scans.
    
    Args:
        scan_list (List[Dict[str, Any]]): List of scans.
        min_scan (int): Index of the minimum scan.
    
    Returns:
        Tuple[plt.Figure, plt.Axes, plt.Axes]: The figure and axes for the plots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 7))
    color = np.random.rand(3,)
    ax1.scatter(scan_list[min_scan]["x"], scan_list[min_scan]["y"], color=color, s=1)
    ax1.scatter(scan_list[min_scan]["pose"][0], scan_list[min_scan]["pose"][1], color=color, s=3)
    ax1.axis([-5.5, 12.5, -12.5, 6.5])
    ax1.set_title('Pose from raw odometry')
    ax2.scatter(scan_list[min_scan]["x"], scan_list[min_scan]["y"], color=color, s=1)
    ax2.scatter(scan_list[min_scan]["pose"][0], scan_list[min_scan]["pose"][1], color=color, s=3)
    ax2.axis([-5.5, 12.5, -12.5, 6.5])
    ax2.set_title('Pose after ICP correction')
    plt.pause(0.1)
    return fig, ax1, ax2

def update_display(ax1, ax2, odom_scan_list, scan_list, scan_index):
    """
    Update the display with the new scan data.
    
    Args:
        ax1 (plt.Axes): The first axis for raw odometry.
        ax2 (plt.Axes): The second axis for ICP corrected pose.
        odom_scan_list (List[Dict[str, Any]]): List of odometry scans.
        scan_list (List[Dict[str, Any]]): List of scans.
        scan_index (int): Index of the current scan.
    """
    color = np.random.rand(3,)
    ax1.scatter(odom_scan_list[scan_index]["x"], odom_scan_list[scan_index]["y"], color=color, s=1)
    ax1.scatter(odom_scan_list[scan_index]["pose"][0], odom_scan_list[scan_index]["pose"][1], color=color, s=3)
    ax2.scatter(scan_list[scan_index]["x"], scan_list[scan_index]["y"], color=color, s=1)
    ax2.scatter(scan_list[scan_index]["pose"][0], scan_list[scan_index]["pose"][1], color=color, s=3)
    plt.pause(0.1)


def main():
    """
    Main function to run the ICP localization demo.
    """
    # Reading data
    try:
        scan_list = datasets.read_u2is(0)
    except Exception as e:
        logging.error(f"Failed to read datasets: {e}")
        return

    # Copy for reference display
    odom_scan_list = copy.deepcopy(scan_list)

    # Parameters for scan processing
    min_scan = 1
    step = 10
    max_scan = len(scan_list) - step

    # Init displays
    fig, ax1, ax2 = initialize_display(odom_scan_list, min_scan)

    for scan_index in range(min_scan, max_scan, step):
        scan1 = scan_list[scan_index]
        scan2 = scan_list[scan_index + step]

        # Perform ICP
        try:
            rotation_matrix, translation_vector, error, iterations = icp.icp(scan1, scan2, 200, 1e-7)
        except Exception as e:
            logging.error(f"ICP failed: {e}")
            continue

        # Correct future scans
        for future_scan_index in range(scan_index + step, max_scan, step):
            scan_list[future_scan_index] = datasets.transform_scan(scan_list[future_scan_index], rotation_matrix, translation_vector)

        # Update display
        update_display(ax1, ax2, odom_scan_list, scan_list, scan_index + step)

if __name__ == "__main__":
    main()