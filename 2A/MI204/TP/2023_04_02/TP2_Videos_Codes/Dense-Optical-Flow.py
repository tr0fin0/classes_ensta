import cv2
import numpy as np

#Ouverture du flux video
cap = cv2.VideoCapture("Extrait1-Cosmos_Laundromat1(340p).m4v")

ret, frame1 = cap.read() # Passe à l'image suivante
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris
hsv = np.zeros_like(frame1) # Image nulle de même taille que frame1 (affichage OF)
hsv[:,:,1] = 255 # Toutes les couleurs sont saturées au maximum

index = 1
ret, frame2 = cap.read()
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

while(ret):
    index += 1
    flow = cv2.calcOpticalFlowFarneback(prvs,next,None, 
                                        pyr_scale = 0.5,# Taux de réduction pyramidal
                                        levels = 3, # Nombre de niveaux de la pyramide
                                        winsize = 15, # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                        iterations = 3, # Nb d'itérations par niveau
                                        poly_n = 7, # Taille voisinage pour approximation polynomiale
                                        poly_sigma = 1.5, # E-T Gaussienne pour calcul dérivées 
                                        flags = 0)	
    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1]) # Conversion cartésien vers polaire
    hsv[:,:,0] = (ang*180)/(2*np.pi) # Teinte (codée sur [0..179] dans OpenCV) <--> Argument
    hsv[:,:,2] = (mag*255)/np.amax(mag) # Valeur <--> Norme 

    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    result = np.vstack((frame2,bgr))
    cv2.imshow('Image et Champ de vitesses (Farnebäck)',result)
    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png'%index,frame2)
        cv2.imwrite('OF_hsv_%04d.png'%index,bgr)
    prvs = next
    ret, frame2 = cap.read()
    if (ret):
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

cap.release()
cv2.destroyAllWindows()
