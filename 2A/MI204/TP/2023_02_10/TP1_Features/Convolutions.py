import numpy as np
import cv2
# from google.colab.patches import cv2_imshow

from matplotlib import pyplot as plt

# 2A/MI204/TP/2023_02_10/TP1_Features

#Lecture image en niveau de gris et conversion en float64
# print("check")
img=np.float64(cv2.imread('../Image_Pairs/FlowerGarden2.png',0))
# print("check")
(h, w) = img.shape
print(f"image : {h} x {w} pixels")
plt.imshow(img,cmap = 'gray')
plt.title('Image Raw')
plt.show()


#Méthode directe
t1 = cv2.getTickCount()
img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(1,h-1):
  for x in range(1,w-1):
    # kernel:
    #    -1
    # -1 +5 -1
    #    -1

    # val = +5*img[y, x] - img[y-1, x] - img[y, x-1] - img[y+1, x] - img[y, x+1] 
    val = -5*img[y, x] + img[y-1, x] + img[y, x-1] + img[y+1, x] + img[y, x+1] 
    img2[y,x] = min(max(val,0),255)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print(f"Méthode Directe : {time:1.8f} s")

# cv2.imshow did not work. gave problems
# cv2.imshow('Avec boucle python',img2.astype(np.uint8))
# # cv2_imshow(img2.astype(np.uint8))
# #Convention OpenCV : une image de type entier est interprétée dans {0,...,255}
# cv2.waitKey()
# cv2.destroyAllWindows()
# print("check")

plt.subplot(121)
plt.imshow(img2,cmap = 'gray')
plt.title('Convolution - Méthode Directe')

#Méthode filter2D
t1 = cv2.getTickCount()
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
img3 = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print(f"Méthode Filter2D: {time:1.8f} s")

# cv2.imshow('Avec filter2D',img3/255.0)
# #Convention OpenCV : une image de type flottant est interprétée dans [0,1]
# cv2.waitKey()

plt.subplot(122)
plt.imshow(img3,cmap = 'gray',vmin = 0.0,vmax = 255.0)
#Convention Matplotlib : par défaut, normalise l'histogramme !
plt.title('Convolution - filter2D')
plt.show()


# Q1
#   comment on peut voir avec le temps d'exécution, l'OpenCV est beaucoup plus efficace.

#   les fonctions pythons sont interpretes par contre que l'OpenCV est compile de la libraire du C++ que rende le code plus efficace.

# Q2
#   on voir que le kernel propose est:
#     kernel:
#     +0 -1 +0
#     -1 +5 -1
#     +0 -1 +0
#   le rehaussement de contraste arrive...
#   f-ftt equations from the board you took from the last class need the math proof

# Q3
#   see poly