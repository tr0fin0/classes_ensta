import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

def cutDetection(videoName: str = "Extrait1-Cosmos_Laundromat1(340p)", maxIndex:int = None):
    # open video
    capture = cv2.VideoCapture(f"../data/{videoName}.m4v")

    if (capture.isOpened() == False):
        print(f'error: {videoName} not opened')
        return

    if maxIndex == None:
        maxIndex = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # read frame of index 0 and convert into gray scale
    index = 0
    ret, frame0 = capture.read()
    prevFrame = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    # read frame of index 1 and convert into gray scale
    index = 1
    ret, frame1 = capture.read()
    nextFrame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # array stores difference between each pair of frames
    total_diffs = []

    while (ret and index < maxIndex):
        index += 1

        diff = cv2.absdiff(nextFrame, prevFrame)
        norm_diff = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        total_diffs.append(cv2.sumElems(norm_diff)[0])

        # configure quit button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # configure save button
        elif (cv2.waitKey(15) & 0xFF) == ord('s'):
            print(f'index: {index}')

        # cv2.imshow('video', prevFrame)

        prevFrame = nextFrame
        ret, frame1 = capture.read()
        if (ret):
            nextFrame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    capture.release()
    cv2.destroyAllWindows()

    # Compute the mean of total_diffs
    mean_diff = np.mean(total_diffs)
    std_diff = np.std(total_diffs)
    threshold = mean_diff + 1.5*std_diff

    diffs = np.abs(np.diff(total_diffs))

    # Plot the graph
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(diffs)
    ax.axhline(y=threshold, color='r', linestyle='--', label='threshold')
    plt.xlabel('frame number')
    plt.ylabel('total difference')
    plt.title(f'{videoName}: cutDetection')

    for i in range(len(diffs)):
        if diffs[i] > threshold:
            ax.plot(i, diffs[i], 'ro')
            ax.text(i, diffs[i], str(i), rotation=(0), bbox=dict(facecolor='white', alpha=0.25))

    plt.legend()
    plt.savefig(f'../images/cut_{videoName}_{maxIndex}.png', dpi=300, bbox_inches='tight')
    plt.show()


cutDetection()

for name in ["Extrait1-Cosmos_Laundromat1(340p)", "Extrait2-ManWithAMovieCamera", "Extrait3-Vertigo-Dream_Scene(320p)", "Extrait4-Entracte-Poursuite_Corbillard(358p)", "Extrait5-Matrix-Helicopter_Scene(280p)"]:
    for size in [None, 500]:
        cutDetection(maxIndex = size, videoName=name)