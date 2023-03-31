import numpy as np
import cv2

cap = cv2.VideoCapture('../Vidéos/Extrait1-Cosmos_Laundromat1(340p).m4v')

# Paramètres du détecteur de points d'intérêt
feature_params = dict( maxCorners = 10000,
                       qualityLevel = 0.01,
                       minDistance = 5,
                       blockSize = 7 )

# Paramètres de l'algo de Lucas et Kanade
lk_params = dict( winSize  = (15,15),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Extraction image initiale et détection des points d'intérêt
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
index = 1

while(ret):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    index += 1

    # Calcul du flot optique
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Sélection des points valides
    good_new = p1[st==1]
    good_old = p0[st==1]
    # Image masque pour tracer les vecteurs de flot
    mask = np.zeros_like(old_frame)

    # Affichage des vecteurs de flot 
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d),(255,255,0),2)
        frame = cv2.circle(frame,(c,d),3,(255,255,0),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('Flot Optique Lucas-Kanade Pyramidal',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('OF_PyrLk%04d.png'%index,img)

    # Mis à jour image et détection des nouveaux points
    p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
    old_gray = frame_gray.copy()

cv2.destroyAllWindows()
cap.release()

