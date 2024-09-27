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

# Global parameters
AXIS_LIMIT = [-5.5, 12.5, -12.5, 6.5] # Axis limits for the u2is dataset
#AXIS_LIMIT = [-20, 25, -10, 30] # Axis limits for the fr079 dataset
DISPLAY = True # Display is very slow, set to False for batch processing

# Parameters for scan processing
MIN_SCAN = 0
STEP = 3

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
    ax1.axis(AXIS_LIMIT)
    ax1.set_title('Pose from raw odometry')
    ax2.scatter(scan_list[min_scan]["x"], scan_list[min_scan]["y"], color=color, s=1)
    ax2.scatter(scan_list[min_scan]["pose"][0], scan_list[min_scan]["pose"][1], color=color, s=3)
    ax2.axis(AXIS_LIMIT)
    ax2.set_title('Pose after ICP correction')
    if DISPLAY:
        plt.pause(0.1)
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
    MAX_SCAN = len(scan_list) - STEP

    # Init displays
    fig, ax1, ax2 = initialize_display(odom_scan_list, MIN_SCAN)

    for scan_index in range(MIN_SCAN, MAX_SCAN, STEP):
        scan1 = scan_list[scan_index]
        scan2 = scan_list[scan_index + STEP]

        # Perform ICP
        try:
            rotation_matrix, translation_vector, error, iterations = icp.icp(scan1, scan2, 200, 1e-7)
        except Exception as e:
            logging.error(f"ICP failed: {e}")
            continue

        # Correct future scans
        for future_scan_index in range(scan_index + STEP, MAX_SCAN, STEP):
            scan_list[future_scan_index] = datasets.transform_scan(scan_list[future_scan_index], rotation_matrix, translation_vector)

        # Update display
        update_display(ax1, odom_scan_list[scan_index + STEP])
        update_display(ax2, scan_list[scan_index + STEP])

    plt.savefig("icpLocalization_Map.png")
    logging.info("Press Q in figure to finish...")
    plt.show()

if __name__ == "__main__":
    main()
