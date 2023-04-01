import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the video
cap = cv2.VideoCapture('Extrait1-Cosmos_Laundromat1(340p).m4v')

# Define the bins for the histogram
bins = [32, 32]

# Define the range for the histogram
range = [[0, 256], [0, 256]]

# Loop through each frame of the video
while True:
    # Read the frame
    ret, frame = cap.read()

    # If no frame is read, break the loop
    if not ret:
        break

    # Convert the frame from RGB to YUV color space
    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)

    # Compute the 2D histogram of the u and v components
    hist, xedges, yedges = np.histogram2d(yuv_frame[:,:,1].ravel(), yuv_frame[:,:,2].ravel(), bins=bins, range=range)

    # Normalize the histogram
    hist = np.log10(hist + 1)
    hist = hist / np.max(hist)

    # Plot the histogram as an image
    plt.imshow(hist, cmap='gray')
    plt.axis('off')
    plt.show()

    # Wait for a key press to display the next frame
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
