import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt



def selectAlgorithm() -> int:
    if len(sys.argv) != 2:
        print("Usage :", sys.argv[0], "detector(= orb ou kaze)")
        sys.exit(2)
    if sys.argv[1].lower() == "orb":
        detector = 1
    elif sys.argv[1].lower() == "kaze":
        detector = 2
    else:
        print("Usage :", sys.argv[0], "detector(= orb ou kaze)")
        sys.exit(2)

    return detector


def Q6():
    img1 = cv2.imread('../Image_Pairs/torb_small1.png')
    img2 = cv2.imread('../Image_Pairs/torb_small2.png')

    start = cv2.getTickCount()

    nFeatures = 250
    kp1_ORB = cv2.ORB_create(
        nfeatures=nFeatures,  #Par défaut : 500
        scaleFactor=2,        #Par défaut : 1.2
        nlevels=3)            #Par défaut : 8
    kp2_ORB = cv2.ORB_create(nfeatures=nFeatures, scaleFactor=2, nlevels=3)

    kp1_KAZE = cv2.KAZE_create(
        upright=False,    #Par défaut : false
        threshold=0.001,  #Par défaut : 0.001
        nOctaves=4,       #Par défaut : 4
        nOctaveLayers=4,  #Par défaut : 4
        diffusivity=2)    #Par défaut : 2
    kp2_KAZE = cv2.KAZE_create(
        upright=False,    #Par défaut : false
        threshold=0.001,  #Par défaut : 0.001
        nOctaves=4,       #Par défaut : 4
        nOctaveLayers=4,  #Par défaut : 4
        diffusivity=2)    #Par défaut : 2

    #Conversion en niveau de gris
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #Détection des keypoints
    pts1_ORB = kp1_ORB.detect(gray1, None)
    pts2_ORB = kp2_ORB.detect(gray2, None)

    pts1_KAZE = kp1_KAZE.detect(gray1, None)
    pts2_KAZE = kp2_KAZE.detect(gray2, None)
    end = cv2.getTickCount()
    time = (end - start) / cv2.getTickFrequency()


    #Affichage des keypoints
    img1_ORB = cv2.drawKeypoints(gray1, pts1_ORB, None, flags=4)
    img2_ORB = cv2.drawKeypoints(gray2, pts2_ORB, None, flags=4)

    img1_KAZE = cv2.drawKeypoints(gray1, pts1_KAZE, None, flags=4)
    img2_KAZE = cv2.drawKeypoints(gray2, pts2_KAZE, None, flags=4)
    # flags définit le niveau d'information sur les points d'intérêt
    # 0 : position seule ; 4 : position + échelle + direction

    plt.subplot(221)
    plt.imshow(img1_ORB)
    plt.title(f'image 1, ORB')

    plt.subplot(222)
    plt.imshow(img2_ORB)
    plt.title(f'image 2, ORB')

    plt.subplot(223)
    plt.imshow(img1_KAZE)
    plt.title(f'image 1, KAZE')

    plt.subplot(224)
    plt.imshow(img2_KAZE)
    plt.title(f'image 2, KAZE')

    plt.savefig(f'./images/Q6A.svg')
    # plt.set_xticks([])
    # plt.set_yticks([])
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()




def Q8():
    def crossCheck():
        img1 = cv2.imread('../Image_Pairs/torb_small1.png')
        img2 = cv2.imread('../Image_Pairs/torb_small2.png')

        start = cv2.getTickCount()

        nFeatures = 250
        kp1_ORB = cv2.ORB_create(nfeatures=nFeatures, scaleFactor=2, nlevels=3)
        kp2_ORB = cv2.ORB_create(nfeatures=nFeatures, scaleFactor=2, nlevels=3)

        kp1_KAZE = cv2.KAZE_create(
            upright=False,    #Par défaut : false
            threshold=0.001,  #Par défaut : 0.001
            nOctaves=4,       #Par défaut : 4
            nOctaveLayers=4,  #Par défaut : 4
            diffusivity=2)    #Par défaut : 2
        kp2_KAZE = cv2.KAZE_create(
            upright=False,    #Par défaut : false
            threshold=0.001,  #Par défaut : 0.001
            nOctaves=4,       #Par défaut : 4
            nOctaveLayers=4,  #Par défaut : 4
            diffusivity=2)    #Par défaut : 2

        #Conversion en niveau de gris
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        #Détection et description des keypoints
        pts1_ORB, desc1_ORB = kp1_ORB.detectAndCompute(gray1, None)
        pts2_ORB, desc2_ORB = kp2_ORB.detectAndCompute(gray2, None)

        pts1_KAZE, desc1_KAZE = kp1_KAZE.detectAndCompute(gray1, None)
        pts2_KAZE, desc2_KAZE = kp2_KAZE.detectAndCompute(gray2, None)

        #Les points non appariés apparaîtront en gris
        img1_ORB = cv2.drawKeypoints(gray1, pts1_ORB, None, color=(127, 127, 127), flags=0)
        img2_ORB = cv2.drawKeypoints(gray2, pts2_ORB, None, color=(127, 127, 127), flags=0)

        img1_KAZE = cv2.drawKeypoints(gray1, pts1_KAZE, None, color=(127, 127, 127), flags=0)
        img2_KAZE = cv2.drawKeypoints(gray2, pts2_KAZE, None, color=(127, 127, 127), flags=0)

        # Calcul de l'appariement
        start = cv2.getTickCount()
        bf_ORB  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        bf_KAZE = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        matches_ORB = bf_ORB.match(desc1_ORB, desc2_ORB)
        matches_ORB = sorted(matches_ORB, key=lambda x: x.distance)

        matches_KAZE = bf_KAZE.match(desc1_KAZE, desc2_KAZE)
        matches_KAZE = sorted(matches_KAZE, key=lambda x: x.distance)

        end = cv2.getTickCount()
        time = (end - start) / cv2.getTickFrequency()
        print(f"{time} s")

        # Trace les N meilleurs appariements
        Nbest = 200
        img3_ORB = cv2.drawMatches(img1_ORB, pts1_ORB, img2_ORB, pts2_ORB, matches_ORB[:Nbest], None, flags=2)
        img3_KAZE = cv2.drawMatches(img1_KAZE, pts1_KAZE, img2_KAZE, pts2_KAZE, matches_KAZE[:Nbest], None, flags=2)

        plt.subplot(121)
        plt.imshow(img3_ORB)
        plt.title(f'Cross Check {Nbest}, ORB')

        plt.subplot(122)
        plt.imshow(img3_KAZE)
        plt.title(f'Cross Check {Nbest}, KAZE')

        plt.savefig(f'./images/Q8CC.svg')
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.show()


    def FLANN():
        img1 = cv2.imread('../Image_Pairs/torb_small1.png')
        img2 = cv2.imread('../Image_Pairs/torb_small2.png')

        start = cv2.getTickCount()

        nFeatures = 250
        kp1_ORB = cv2.ORB_create(nfeatures=nFeatures, scaleFactor=2, nlevels=3)
        kp2_ORB = cv2.ORB_create(nfeatures=nFeatures, scaleFactor=2, nlevels=3)

        kp1_KAZE = cv2.KAZE_create(
            upright=False,    #Par défaut : false
            threshold=0.001,  #Par défaut : 0.001
            nOctaves=4,       #Par défaut : 4
            nOctaveLayers=4,  #Par défaut : 4
            diffusivity=2)    #Par défaut : 2
        kp2_KAZE = cv2.KAZE_create(
            upright=False,    #Par défaut : false
            threshold=0.001,  #Par défaut : 0.001
            nOctaves=4,       #Par défaut : 4
            nOctaveLayers=4,  #Par défaut : 4
            diffusivity=2)    #Par défaut : 2

        #Conversion en niveau de gris
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        #Détection et description des keypoints
        pts1_ORB, desc1_ORB = kp1_ORB.detectAndCompute(gray1, None)
        pts2_ORB, desc2_ORB = kp2_ORB.detectAndCompute(gray2, None)

        desc1_ORB = np.float64(desc1_ORB)
        desc2_ORB = np.float64(desc2_ORB)
        # if(desc1_ORB.dtype()!=CV_32F):
        #     desc1_ORB.convertTo(desc1_ORB, CV_32F)

        # if(desc2_ORB.dtype()!=CV_32F):
        #     desc2_ORB.convertTo(desc2_ORB, CV_32F)


        pts1_KAZE, desc1_KAZE = kp1_KAZE.detectAndCompute(gray1, None)
        pts2_KAZE, desc2_KAZE = kp2_KAZE.detectAndCompute(gray2, None)

        # Calcul de l'appariement
        start = cv2.getTickCount()
        FLANN_INDEX_KDTREE = 0
        index_params  = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches_ORB  = flann.knnMatch(desc1_ORB, desc2_ORB, k=2)
        matches_KAZE = flann.knnMatch(desc1_KAZE, desc2_KAZE, k=2)

        # Application du ratio test
        good_ORB = []
        for m,n in matches_ORB:
          if m.distance < 0.7*n.distance:
            good_ORB.append([m])

        good_KAZE = []
        for m,n in matches_KAZE:
          if m.distance < 0.7*n.distance:
            good_KAZE.append([m])

        draw_params = dict(matchColor = (0,255,0),
                          singlePointColor = (255,0,0),
                          flags = 0)

        # Affichage des appariements qui respectent le ratio test
        img3_ORB = cv2.drawMatchesKnn(gray1, pts1_ORB, gray2, pts2_ORB, good_ORB, None, **draw_params)
        img3_KAZE = cv2.drawMatchesKnn(gray1, pts1_KAZE, gray2, pts2_KAZE, good_KAZE, None, **draw_params)

        end = cv2.getTickCount()
        time = (end - start) / cv2.getTickFrequency()
        print(f"{time} s")

        plt.subplot(121)
        plt.imshow(img3_ORB)
        plt.title(f'FLANN {len(good_ORB)}, ORB')

        plt.subplot(122)
        plt.imshow(img3_KAZE)
        plt.title(f'FLANN {len(good_KAZE)}, KAZE')

        plt.savefig(f'./images/Q8FL.svg')
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.show()


    def RATIO():
        img1 = cv2.imread('../Image_Pairs/torb_small1.png')
        img2 = cv2.imread('../Image_Pairs/torb_small2.png')

        nFeatures = 250
        kp1_ORB = cv2.ORB_create(nfeatures=nFeatures, scaleFactor=2, nlevels=3)
        kp2_ORB = cv2.ORB_create(nfeatures=nFeatures, scaleFactor=2, nlevels=3)

        kp1_KAZE = cv2.KAZE_create(
            upright=False,    #Par défaut : false
            threshold=0.001,  #Par défaut : 0.001
            nOctaves=4,       #Par défaut : 4
            nOctaveLayers=4,  #Par défaut : 4
            diffusivity=2)    #Par défaut : 2
        kp2_KAZE = cv2.KAZE_create(
            upright=False,    #Par défaut : false
            threshold=0.001,  #Par défaut : 0.001
            nOctaves=4,       #Par défaut : 4
            nOctaveLayers=4,  #Par défaut : 4
            diffusivity=2)    #Par défaut : 2

        gray1 =  cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 =  cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        pts1_ORB, desc1_ORB = kp1_ORB.detectAndCompute(gray1, None)
        pts2_ORB, desc2_ORB = kp2_ORB.detectAndCompute(gray2, None)

        pts1_KAZE, desc1_KAZE = kp1_KAZE.detectAndCompute(gray1, None)
        pts2_KAZE, desc2_KAZE = kp2_KAZE.detectAndCompute(gray2, None)


        start = cv2.getTickCount()

        bf_ORB  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        bf_KAZE = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        matches_ORB = bf_ORB.knnMatch(desc1_ORB, desc2_ORB, k=2)
        matches_KAZE = bf_KAZE.knnMatch(desc1_KAZE, desc2_KAZE, k=2)

        good_ORB = []
        for m,n in matches_ORB:
            if m.distance < 0.7*n.distance:
                good_ORB.append([m])

        good_KAZE = []
        for m,n in matches_KAZE:
            if m.distance < 0.7*n.distance:
                good_KAZE.append([m])


        end = cv2.getTickCount()
        time = (end - start)/ cv2.getTickFrequency()

        # Affichage des appariements qui respectent le ratio test
        draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        flags = 0)
        img3_ORB = cv2.drawMatchesKnn(gray1, pts1_ORB, gray2, pts2_ORB, good_ORB, None, **draw_params)
        img3_KAZE = cv2.drawMatchesKnn(gray1, pts1_KAZE, gray2, pts2_KAZE, good_KAZE, None, **draw_params)

        plt.subplot(121)
        plt.imshow(img3_ORB)
        plt.title(f'Ratio Test {len(good_ORB)}, ORB')

        plt.subplot(122)
        plt.imshow(img3_KAZE)
        plt.title(f'Ratio Test {len(good_KAZE)}, KAZE')

        plt.savefig(f'./images/Q8RT.svg')
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.show()


    crossCheck()
    # FLANN()
    # RATIO()




def main():
    # Q6()
    Q8()





if __name__ == "__main__":
    main()