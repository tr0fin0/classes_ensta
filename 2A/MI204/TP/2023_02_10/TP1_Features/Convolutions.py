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
      val = kernel[0][0]*image[y+1, x-1] + kernel[0][1]*image[y+1, x+0] + kernel[0][2]*image[y+1, x+1] + kernel[1][0]*image[y+0, x-1] + kernel[1][1]*image[y+0, x+0] + kernel[1][2]*image[y+0, x+1] +kernel[2][0]*image[y-1, x-1] + kernel[2][1]*image[y-1, x+0] + kernel[2][2]*image[y-1, x+1]

      # for i in [-1, 0, +1]:
      #   for j in [-1, 0, +1]:
          # val += kernel[0,0]*image[y+1, x-1] + kernel[0,1]*image[y+1, x+0] + kernel[0,2]*image[y+1, x+1] + kernel[1,0]*image[y+0, x-1] + kernel[1,1]*image[y+0, x+0] + kernel[1,2]*image[y+0, x+1] +kernel[2,0]*image[y-1, x-1] + kernel[2,1]*image[y-1, x+0] + kernel[2,2]*image[y-1, x+1]
      # image[y-1, x] - image[y, x-1] - image[y+1, x] - image[y, x+1] 
      # val = +5*image[y, x] - image[y-1, x] - image[y, x-1] - image[y+1, x] - image[y, x+1] 

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


def methodOpenCV(image) -> np.float64:
  """
  methodOpenCV()

  """

  #Méthode filter2D
  start = cv2.getTickCount()
  kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
  img = cv2.filter2D(image, -1, kernel)
  end = cv2.getTickCount()

  time = (start - end)/ cv2.getTickFrequency()
  print(f"Méthode Filter2D: {time:1.8f} s")
  # il faut normalizer l'image
  plotImg(image, 'Convolution - filter2D', 0.0, 255.0)

  return img



def main() -> None:
  image = np.float64(cv2.imread('../Image_Pairs/FlowerGarden2.png',0))

  plotImg(image, 'image original')

  imageDiscrete = methodDiscrete(image)
  imageOpenCV = methodOpenCV(image)

  return None






if __name__ == "__main__":
  main()


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