import numpy as np
import cv2
from matplotlib import pyplot as plt

# 2A/MI204/TP/2023_02_10/TP1_Features




def plotImg(image, title: str = 'title', vmin: float = -1.0, vmax: float = -1.0) -> None:
  """
  plotImg():

  """

  if vmin == -1 or vmax == -1:
    plt.imshow(image, cmap = 'gray')
  else:
    plt.imshow(image, cmap = 'gray', vmin = vmin, vmax = vmax)

  plt.title(title)
  plt.show()
  
  return None


def convolution(image, kernel) -> np.float64:
  img = cv2.copyMakeBorder(image,0,0,0,0,cv2.BORDER_REPLICATE)
  (h, w) = img.shape

  for x in range(1,w-1):
    for y in range(1,h-1):
      val = 0
      for j in range(kernel.shape[0]):
        for i in range(kernel.shape[1]):
          val += kernel[i, j]*image[y+1-i, x-1+j]

      img[y,x] = min(max(val,0),255)

  return img


def methodDiscrete(image, showImage: bool = True) -> np.float64:
  """
  methodDiscrete():

  """

  #Méthode directe
  start = cv2.getTickCount()
  kernel= [
    [+0, -1, +0],
    [-1, +5, -1],
    [+0, -1, +0]
  ]
  img = convolution(image, kernel)
  end = cv2.getTickCount()

  time = (end - start)/ cv2.getTickFrequency()
  print(f"Méthode Directe : {time:1.4e} s")

  if showImage:
    plotImg(img, 'Convolution - Méthode Directe')

  return img


def methodOpenCV(image, showImage: bool = True) -> np.float64:
  """
  methodOpenCV()

  """

  #Méthode filter2D
  start = cv2.getTickCount()
  kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
  img = cv2.filter2D(image, -1, kernel)
  end = cv2.getTickCount()

  time = (end-start)/ cv2.getTickFrequency()
  print(f"Méthode Filter2D: {time:1.4e} s")
  # il faut normalizer l'image
  if showImage:
    plotImg(image, 'Convolution - filter2D', 0.0, 255.0)

  return img


def Q1() -> None:
  """
  Q1()

  """
  print("[Q1]")
  image = np.float64(cv2.imread('../Image_Pairs/FlowerGarden2.png',0))

  imageDiscrete = methodDiscrete(image, False)
  imageOpenCV = methodOpenCV(image, False)


  figure, axis = plt.subplots(2, 2)

  axis[0, 0].imshow(image, cmap = 'gray')
  axis[0, 0].set_title("original")
  axis[1, 0].imshow(imageDiscrete, cmap = 'gray')
  axis[1, 0].set_title("discrete")
  axis[1, 1].imshow(imageOpenCV, cmap = 'gray', vmin = 0.0, vmax = 255.0)
  axis[1, 1].set_title("openCV")
  plt.show()

  return None


def Q3() -> None:
  """
  Q3()

  """
  print("[Q3]")
  image = np.float64(cv2.imread('../Image_Pairs/FlowerGarden2.png',0))

  start = cv2.getTickCount()
  hx = [
      [-1, +0, +1],
      [-2, +0, +2],
      [-1, +0, +1],
    ]
  hy = [
      [-1, -2, -1],
      [+0, +0, +0],
      [+1, +2, +1],
    ]

  fx = convolution(image, hx)
  fy = convolution(image, hy)

  img = (fx**2 + fy**2)**(1/2)
  end = cv2.getTickCount()

  time = (end-start)/ cv2.getTickFrequency()
  print(f"\tNorme Gradient Euclidienne: {time:1.4e} s")


  figure, axis = plt.subplots(2, 2)

  axis[0, 0].imshow(image, cmap = 'gray')
  axis[0, 0].set_title("original")
  axis[0, 1].imshow(img, cmap = 'gray', vmin = 0.0, vmax = 255.0)
  axis[0, 1].set_title("gradient euclidienne")
  axis[1, 0].imshow(fx, cmap = 'gray', vmin = 0.0, vmax = 255.0)
  axis[1, 0].set_title("derivate x")
  axis[1, 1].imshow(fy, cmap = 'gray', vmin = 0.0, vmax = 255.0)
  axis[1, 1].set_title("derivate y")
  plt.show()

  return None




def main() -> None:
  # Q1()
  Q3()

  return None






if __name__ == "__main__":
  main()


  # Q1
  #   comment on peut voir avec le temps d'exécution, l'OpenCV est beaucoup plus efficace.
  #     OpenCV:   4.2532e-04s
  #     Discrete: 5.6706e-02s
  #   tous les images ont êtes imprimées ensemble pour la comparaison visuelle.

  #   les fonctions pythons sont interpretes par contre que l'OpenCV est compile de la libraire du C++ que rende le code plus efficace pendant l'exécution.


  # Q2
  #   on voir que le kernel propose est:
  #     kernel:
  #     +0 -1 +0
  #     -1 +5 -1
  #     +0 -1 +0
  #   le rehaussement de contraste arrive...
  #   f-ftt equations from the board you took from the last class need the math proof


  # Q3
  #   À partir de la page 56 du slide "1 Introduction Modeles" on peut voir que les composantes du gradient peut être faites avec la convolution des Masques de Sobel données pour les Kernels hx et hy suivants:

  #   fx[i,j] = (f * hx)[i,j]
  #     hx = [
  #         [-1, +0, +1],
  #         [-2, +0, +2],
  #         [-1, +0, +1],
  #       ]
  #   fj[i,j] = (f * hj)[i,j]
  #     hy = [
  #         [-1, -2, -1],
  #         [+0, +0, +0],
  #         [+1, +2, +1],
  #       ]

  #   Après la Norme Euclidienne du Gradient, https://en.wikipedia.org/wiki/Norm_(mathematics), aussi connu comment la Norme 2, peut être obitenu avec l'equation suivante:
  #   ||Df||2 = sqrt(fx^2 + fy^2)

  # Pour l'affichage correcte il faudrait normalizer les valeurs de l'image en modifiant les variables "vmin" et "vmax" sur la fonction ".imshow()"