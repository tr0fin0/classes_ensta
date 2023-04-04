import cv2
import numpy as np

def opticalFlow(
    pyr_scale=0.5, 
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=7,
    poly_sigma=1.5,
    flags=0
    ):
    #Ouverture du flux video
    videoName = "Extrait1-Cosmos_Laundromat1(340p)"
    # videoName = "Extrait2-ManWithAMovieCamera"
    # videoName = "Extrait3-Vertigo-Dream_Scene(320p)"
    # videoName = "Extrait4-Entracte-Poursuite_Corbillard(358p)"
    # videoName = "Extrait5-Matrix-Helicopter_Scene(280p)"
    # videoName = "Rotation_OX(Tilt)"
    # videoName = "Rotation_OY(Pan)"
    # videoName = "Rotation_OZ(Roll)"
    # videoName = "Travelling_OX"
    # videoName = "Travelling_OZ"
    # videoName = "ZOOM_O_TRAVELLING"
    capture = cv2.VideoCapture(f"../data/{videoName}.m4v")

    if (capture.isOpened() == False):
        print(f'error')

    index = 0
    ret, frame0 = capture.read()                            # Passe à l'image suivante
    previousFrame = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)# Passage en niveaux de gris
    hsv = np.zeros_like(frame0)                             # Image nulle de même taille que frame0 (affichage OF)
    hsv[:, :, 1] = 255                                      # Toutes les couleurs sont saturées au maximum

    index = 1
    ret, frame1 = capture.read()
    nextFrame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # execute in the whole video
    while (ret):
        index += 1

        flow = cv2.calcOpticalFlowFarneback(previousFrame, nextFrame, None,
            pyr_scale=pyr_scale,  # Taux de réduction pyramidal
            levels=levels,       # Nombre de niveaux de la pyramide
            winsize=winsize,     # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
            iterations=iterations,   # Nb d'itérations par niveau
            poly_n=poly_n,       # Taille voisinage pour approximation polynomiale
            poly_sigma=poly_sigma, # E-T Gaussienne pour calcul dérivées 
            flags=flags)

        # Conversion cartésien vers polaire
        mag, ang = cv2.cartToPolar(flow[:, :, 0],
                                flow[:, :, 1])
        # Teinte (codée sur [0..179] dans OpenCV) <--> Argument
        hsv[:, :, 0] = (ang * 180) / (2 * np.pi)
        hsv[:, :, 2] = (mag * 255) / np.amax(mag)  # Valeur <--> Norme

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result = np.vstack((frame1, bgr))
        cv2.imshow('Farneback Optical Flow', result)

        # configure quit button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # configure save frame of video and optical flow
        elif (cv2.waitKey(15) & 0xFF) == ord('s'):
            cv2.imwrite(f'../images/frame_{index}_{videoName}.png', frame1)
            cv2.imwrite(f'../images/frame_{index}_{videoName}_opticalFlow.png', bgr)

        # if index == 60:
        #     cv2.imwrite(f'../images/frame_{index}_{videoName}_{pyr_scale}_{levels}_{winsize}_{iterations}_{poly_n}_{poly_sigma}_{flags}.png', np.vstack((frame1, bgr)))
        #     break

        previousFrame = nextFrame
        ret, frame1 = capture.read()
        if (ret):
            nextFrame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    capture.release()
    cv2.destroyAllWindows()

opticalFlow(0.2,1,5,1,5,1.5,0)