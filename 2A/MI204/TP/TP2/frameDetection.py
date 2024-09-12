import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy


def frameDetection(videoName: str = "Extrait1-Cosmos_Laundromat1(340p)", maxIndex:int = None):
    # open video
    capture = cv2.VideoCapture(f"../data/{videoName}.m4v")

    if (capture.isOpened() == False):
        print(f'error: {videoName} not opened')
        return

    if maxIndex == None:
        maxIndex = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    index = 0
    ret, frame = capture.read()

    # array stores difference between each pair of frames
    entropys = []

    while (ret and index < maxIndex):
        index += 1

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute the histogram of the grayscale frame
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        # Normalize the histogram
        hist_norm = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Compute the entropy of the normalized histogram
        entropy = np.sum(-hist_norm*np.log2(hist_norm + (hist_norm==0) + 1e-8))
        entropys.append(entropy)

        # configure quit button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # configure save button
        elif (cv2.waitKey(15) & 0xFF) == ord('s'):
            print(f'index: {index}')

        ret, frame = capture.read()


    capture.release()
    cv2.destroyAllWindows()

    # Compute the mean of entropys
    diffs = np.abs(np.diff(entropys))

    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    threshold = mean_diff + 1.5*std_diff

    # Plot the graph
    fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(entropys)
    ax.plot(diffs)
    ax.axhline(y=threshold, color='r', linestyle='--', label='threshold')
    plt.xlabel('frame number')
    plt.ylabel('total difference')
    plt.title(f'{videoName}: frameDetection')

    for i in range(len(diffs)):
        if diffs[i] > threshold:
            ax.plot(i, diffs[i], 'ro')
            ax.text(i, diffs[i], str(i), rotation=(0), bbox=dict(facecolor='white', alpha=0.25))

    plt.legend()
    plt.savefig(f'../images/frameDetection_{videoName}_{maxIndex}.png', dpi=300, bbox_inches='tight')
    plt.show()


frameDetection()

for name in ["Extrait1-Cosmos_Laundromat1(340p)", "Extrait2-ManWithAMovieCamera", "Extrait3-Vertigo-Dream_Scene(320p)", "Extrait4-Entracte-Poursuite_Corbillard(358p)", "Extrait5-Matrix-Helicopter_Scene(280p)"]:
    for size in [None, 500]:
        frameDetection(maxIndex = size, videoName=name)