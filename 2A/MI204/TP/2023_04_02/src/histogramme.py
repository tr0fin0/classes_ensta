import cv2
import matplotlib.pyplot as plt
import numpy as np

Comp_hist_list1 = []
Comp_hist_list2 = []
Comp_hist_list3 = []
Frame_list = []
Frame=0

cap = cv2.VideoCapture("C:/Users/Vane/OneDrive - Universidad EIA/ENSTA - Paris/Analyse et Indexation Images/TP2/TP2_Videos/Extrait1-Cosmos_Laundromat1(340p).m4v")
ret, img = cap.read()

yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
Prehist = cv2.calcHist([yuv], [1, 2], None, [256, 256], [0, 256, 0, 256])
Prehist = cv2.normalize(Prehist, None, 0, 1, cv2.NORM_MINMAX) #Normalizar entre 0 y 1

#yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) # Y - Brillo, U - Azul, V - Rojo

#Histograma es la frecuencia con la que cambian las imagenes en un video (Frecuencia de la variable que esta representando)
#Hist = cv2.calcHist(yuv, [1, 2], None, [256, 256], [0, 256, 0, 256]) #cv2.calcHist(images, channels, mask, histSize, ranges) 
#Channels: Para sacar los componentes u y v
#Pas mask porque la mascara es si estuvieramos interesados en una parte especifica de la imagen
#histSize: number of bins we want to use when computing a histogram (Cantidad de barras en el histograma)
#ranges: The range of possible pixel values.

while cap.isOpened():
    ret, img = cap.read() 

    if ret == True:
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) # Y - Brillo, U - Azul, V - Rojo
        Hist = cv2.calcHist(yuv, [1, 2], None, [256, 256], [0, 256, 0, 256])
        Hist = cv2.normalize(Hist, None, 0, 1, cv2.NORM_MINMAX) #Normalizar entre 0 y 1

        Comp_hist1 = cv2.compareHist(Hist, Prehist, cv2.HISTCMP_CORREL)
        Comp_hist2 = cv2.compareHist(Hist, Prehist, cv2.HISTCMP_CHISQR)
        Comp_hist3 = cv2.compareHist(Hist, Prehist, cv2.HISTCMP_BHATTACHARYYA)

        Comp_hist_list1.append(Comp_hist1)
        Comp_hist_list2.append(Comp_hist2)
        Comp_hist_list3.append(Comp_hist3)
        
        Frame_list.append(Frame)
        Prehist = Hist
        Frame+=1

        cv2.imshow('video',img)
        cv2.waitKey(40)

    else:
        break


plt.subplot(3, 1, 1)
plt.plot(Frame_list, Comp_hist_list1)
plt.ylabel("Methode Corrélation")
plt.title('Comparaison des histogrammes')

plt.subplot(3, 1, 2)
plt.plot(Frame_list, Comp_hist_list2)
plt.ylabel('Methode Loi du χ²')

plt.subplot(3, 1, 3)
plt.plot(Frame_list, Comp_hist_list3)
plt.xlabel('Frame Number')
plt.ylabel('Methode Bhattacharyya')


plt.show()

cap.release()
cv2.destroyAllWindows()
