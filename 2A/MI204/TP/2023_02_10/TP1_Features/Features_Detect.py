import numpy as np
import cv2

from matplotlib import pyplot as plt

import sys
if len(sys.argv) != 2:
  print ("Usage :",sys.argv[0],"detector(= orb ou kaze)")
  sys.exit(2)
if sys.argv[1].lower() == "orb":
  detector = 1
elif sys.argv[1].lower() == "kaze":
  detector = 2
else:
  print ("Usage :",sys.argv[0],"detector(= orb ou kaze)")
  sys.exit(2)

#Lecture de la paire d'images
img1 = cv2.imread('../Image_Pairs/torb_small1.png')
print("Dimension de l'image 1 :",img1.shape[0],"lignes x",img1.shape[1],"colonnes")
print("Type de l'image 1 :",img1.dtype)
img2 = cv2.imread('../Image_Pairs/torb_small2.png')
print("Dimension de l'image 2 :",img2.shape[0],"lignes x",img2.shape[1],"colonnes")
print("Type de l'image 2 :",img2.dtype)

#Début du calcul
t1 = cv2.getTickCount()
#Création des objets "keypoints"
if detector == 1:
  kp1 = cv2.ORB_create(nfeatures = 250,#Par défaut : 500
                       scaleFactor = 2,#Par défaut : 1.2
                       nlevels = 3)#Par défaut : 8
  kp2 = cv2.ORB_create(nfeatures=250,
                        scaleFactor = 2,
                        nlevels = 3)
  print("Détecteur : ORB")
else:
  kp1 = cv2.KAZE_create(upright = False,#Par défaut : false
    		        threshold = 0.001,#Par défaut : 0.001
  		        nOctaves = 4,#Par défaut : 4
		        nOctaveLayers = 4,#Par défaut : 4
		        diffusivity = 2)#Par défaut : 2
  kp2 = cv2.KAZE_create(upright = False,#Par défaut : false
	  	        threshold = 0.001,#Par défaut : 0.001
		        nOctaves = 4,#Par défaut : 4
		        nOctaveLayers = 4,#Par défaut : 4
		        diffusivity = 2)#Par défaut : 2
  print("Détecteur : KAZE")
#Conversion en niveau de gris
gray1 =  cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 =  cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#Détection des keypoints
pts1 = kp1.detect(gray1,None)
pts2 = kp2.detect(gray2,None)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Détection des points d'intérêt :",time,"s")

#Affichage des keypoints
img1 = cv2.drawKeypoints(gray1, pts1, None, flags=4)
# flags définit le niveau d'information sur les points d'intérêt
# 0 : position seule ; 4 : position + échelle + direction
img2 = cv2.drawKeypoints(gray2, pts2, None, flags=4)

plt.subplot(121)
plt.imshow(img1)
plt.title('Image n°1')

plt.subplot(122)
plt.imshow(img2)
plt.title('Image n°2')

plt.show()


# detection de harris 
# page 28 2_caracteristiques

# page 34 2_caracteristiques
# EVALUATION DES DETECTEURS DE POINTS D’INTERET
# La plupart des détecteurs de point d’intérêt sont définis indépendamment des descripteurs
# avec lesquels on les utilise. Il est donc nécessaire de pouvoir les évaluer en eux-mêmes.
# Les propriétés recherchées d’un bon détecteur :
# • Répétabilité : le point doit
# apparaître aux mêmes endroits
# quelque soit la déformation.
# • Représentativité : les points
# doivent être le plus nombreux
# possible.
# • Efficacité : le détecteur doit être
# rapide à calculer (cf SURF, FAST)
# (Rq : répétabilité et représentativité ne
# sont pas indépendants !)

# documentation of the openCV library to better understand the different codes avaleables the video is an interpretation of the text
#   https://docs.opencv.org/3.4/df/d54/tutorial_py_features_meaning.html
#     https://www.youtube.com/watch?v=DZtUt4bKtmY
# 

# introduction of the facts with the wikipedia page to explain what is the features detection method
#   https://en.wikipedia.org/wiki/Feature_(computer_vision)#Detectors

# sobel kernels are the a kind of feature detection, an edge detection features


# paper that explains in its abstract the differences in the code of the laze and in the orb explaining why it is beeter in some ways
#   https://www.doc.ic.ac.uk/~ajd/Publications/alcantarilla_etal_eccv2012.pdf