"""
Test ICP localisation
Apply a random displacement to a scan and check the error of the recovered position through ICP
author: David Filliat
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import readDatasets as datasets
import icp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
NUM_TESTS = 10
SCAN_INDEX = 55 # Scan to use in the u2is dataset
RANDOM_TRANSLATION_MAX = 1
RANDOM_ROTATION_MAX = 0.6
ICP_MAX_ITERATIONS = 200
ICP_TOLERANCE = 1e-7
DISPLAY = True

def main():
    """
    Main function to test ICP localization.
    Args:
        None
    Returns:
        mean_translation_error: mean translation error
        var_translation_error: variance of translation error
        mean_rotation_error: mean rotation error
        var_rotation_error: variance of rotation error
        mean_time: mean computation time
    """

    # Reading some data
    scan_list = datasets.read_u2is(SCAN_INDEX+1)
    scan_original = scan_list[SCAN_INDEX]
    scan_true_pose = np.array([0.3620, 0.0143, 0.0483])  # Manual estimation for scan 55 of u2is dataset

    # Initialise error log
    pose_error = np.zeros((3, NUM_TESTS))

    time_start = time.process_time()
    for test_index in range(NUM_TESTS):
        try:
            ref_scan_index = np.random.randint(SCAN_INDEX-5)
            ref_scan = scan_list[ref_scan_index]

            # Generate random displacement and apply it to the second scan
            random_translation = np.random.rand(2, 1)*RANDOM_TRANSLATION_MAX -RANDOM_TRANSLATION_MAX/2
            random_rotation = RANDOM_ROTATION_MAX * np.random.rand(1, 1).item() - RANDOM_ROTATION_MAX/2
            rotation_matrix = np.array([
                [math.cos(random_rotation), -math.sin(random_rotation)],
                [math.sin(random_rotation), math.cos(random_rotation)]
            ])
            displaced_scan = datasets.transform_scan(scan_original, rotation_matrix, random_translation)

            if DISPLAY:
                # Display initial positions
                plt.cla()
                plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(ref_scan["x"], ref_scan["y"], "ob", label='Ref Scan')
                plt.plot(displaced_scan["x"], displaced_scan["y"], ".r", label='Scan before ICP')
                plt.axis("equal")

            # Perform ICP
            rotation_matrix, translation_vector, error, iter = icp.icp(ref_scan, displaced_scan, ICP_MAX_ITERATIONS, ICP_TOLERANCE)

            # Apply motion to scan
            transformed_scan = datasets.transform_scan(displaced_scan, rotation_matrix, translation_vector)
            pose_error[:, test_index] = np.transpose(transformed_scan["pose"] - scan_true_pose)

            if DISPLAY:
                # Display
                plt.axis("equal")
                plt.plot(transformed_scan["x"], transformed_scan["y"], ".g", label='Scan after ICP')
                plt.legend()
                plt.pause(0.1)
        except Exception as e:
            print(f"Error during test {test_index}: {e}")

    elapsed_time = time.process_time() - time_start
    translation_errors = np.sqrt(np.square(pose_error[0, :]) + np.square(pose_error[1, :]))
    rotation_errors = np.sqrt(np.square(pose_error[2, :]))

    mean_translation_error = np.mean(translation_errors)
    var_translation_error = np.var(translation_errors)
    mean_rotation_error = np.mean(rotation_errors)
    var_rotation_error = np.var(rotation_errors)
    mean_time = elapsed_time / NUM_TESTS

    # Log summary
    logging.info("Summary of ICP Tests")
    logging.info("--------------------")
    logging.info(f"Mean (var) translation error: {mean_translation_error:.3e} ({var_translation_error:.3e})")
    logging.info(f"Mean (var) rotation error: {mean_rotation_error:.3e} ({var_rotation_error:.3e})")
    logging.info(f"Mean computation time: {mean_time:.4f} seconds")
    logging.info("Press Q in figure to finish...")

    plt.show()

    return mean_translation_error,var_translation_error,mean_rotation_error,var_rotation_error,mean_time

if __name__ == "__main__":
    main()