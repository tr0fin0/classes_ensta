# https://www.geekering.com/programming-languages/python/brunorsilva/harris-corner-detector-python/
import cv2
import numpy as np
from matplotlib import pyplot as plt


def cornerDetectionHarris(imagePath: str, W: float, alpha: float, sigma: int, threshold):
    """
    Harris

        Input:
            -imagePath: path to the image
            -W:         block size
            -alpha:     Harris detector free parameter
            -threshold: Aperture parameter of the Sobel derivative
    """


    image = cv2.imread(imagePath)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageGaussian = cv2.GaussianBlur(imageGray, (3, 3), sigma)

    if image is None:
        print(f'error: invalid image: {imagePath}')
        return None

    height = image.shape[0] #.shape[0] outputs height
    width  = image.shape[1] #.shape[1] outputs width
    matrixTheta = np.zeros((height, width))


    dx = cv2.Sobel(imageGaussian, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(imageGaussian, cv2.CV_64F, 0, 1, ksize=3)

    dx2 = np.square(dx)
    dy2 = np.square(dy)
    dxy = dx * dy


    offset = int(W / 2)
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            Sx2 = np.sum(dx2[y - offset: y + 1 + offset,
                             x - offset: x + 1 + offset])
            Sy2 = np.sum(dy2[y - offset: y + 1 + offset,
                             x - offset: x + 1 + offset])
            Sxy = np.sum(dxy[y - offset: y + 1 + offset,
                             x - offset: x + 1 + offset])

            H = np.array([[Sx2, Sxy], [Sxy, Sy2]])

            # matrix Theta
            T = np.linalg.det(H) - alpha * (np.matrix.trace(H))**2
            matrixTheta[y - offset, x - offset] = T


    cv2.normalize(matrixTheta, matrixTheta, 0, 1, cv2.NORM_MINMAX)
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            value = matrixTheta[y, x]

            if value > threshold:
                cv2.circle(image, (x, y), 1, (255, 0, 0))

    return image


def openCVHarris(imagePath: str, W: float, alpha: float, threshold):
    image = cv2.imread(imagePath)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    harris = cv2.cornerHarris(imageGray, W, threshold, alpha)

    # Result is dilated for marking the corners, not important
    harris = cv2.dilate(harris, None)

    # Threshold for an optimal value, it may vary depending on the image.
    image[harris > 0.01 * harris.max()] = [0, 0, 255]

    return image


def Q4():
    imgPath = "../Image_Pairs/Graffiti0.png"


    def evaluateBlockSize():
        variables = [3, 6, 9, 12]

        figure, axis = plt.subplots(1, len(variables))
        for i in range(len(variables)):
            W = variables[i]
            A = 0.006
            S = 0
            imgManual = cornerDetectionHarris(imgPath, W, A, S, 0.1)

            axis[i].imshow(imgManual, cmap='gray', vmin=0.0, vmax=255.0)
            axis[i].set_title(f'W {W}, A {A}, S {S}')
            axis[i].set_xticks([])
            axis[i].set_yticks([])

        W = 1
        A = 0
        S = 0
        plt.savefig(f'./images/Q5{W}{A}{S}.svg')
        plt.show()


    def evaluateAlpha():
        variables = [0.001, 0.006, 0.007, 0.008]

        figure, axis = plt.subplots(1, len(variables))
        for i in range(len(variables)):
            W = 3
            A = variables[i]
            S = 0
            imgManual = cornerDetectionHarris(imgPath, W, A, S, 0.1)

            axis[i].imshow(imgManual, cmap='gray', vmin=0.0, vmax=255.0)
            axis[i].set_title(f'W {W}, A {A}, S {S}')
            axis[i].set_xticks([])
            axis[i].set_yticks([])

        W = 0
        A = 1
        S = 0
        plt.savefig(f'./images/Q5{W}{A}{S}.svg')
        plt.show()


    def evaluateSigma():
        variables = [1, 2, 3, 5]

        figure, axis = plt.subplots(1, len(variables))
        for i in range(len(variables)):
            W = 3
            A = 0.006
            S = variables[i]
            imgManual = cornerDetectionHarris(imgPath, W, A, S, 0.1)

            axis[i].imshow(imgManual, cmap='gray', vmin=0.0, vmax=255.0)
            axis[i].set_title(f'W {W}, A {A}, S {S}')
            axis[i].set_xticks([])
            axis[i].set_yticks([])

        W = 0
        A = 0
        S = 1
        plt.savefig(f'./images/Q5{W}{A}{S}.svg')
        plt.show()


    evaluateBlockSize()
    evaluateAlpha()
    evaluateSigma()







def main():
    Q4()




if __name__ == "__main__":
    main()