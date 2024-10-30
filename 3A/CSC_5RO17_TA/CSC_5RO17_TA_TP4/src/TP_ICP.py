# TP MAREVA Nuages de Points et Mod�lisation 3D - Python - FG 24/09/2020
# coding=utf8

# Import Numpy
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from mpl_toolkits.mplot3d import Axes3D

# Import functions from scikit-learn : KDTree
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply
# utils.ply est le chemin relatif utils/ply.py


def read_data_ply(path):
# Lecture de nuage de points sous format ply
    '''
    Lecture de nuage de points sous format ply
    Inputs :
        path = chemin d'acc�s au fichier
    Output :
        data = matrice (3 x n)
    '''
    data_ply = read_ply(path)
    data = np.vstack((data_ply['x'], data_ply['y'], data_ply['z']))
    return(data)

def write_data_ply(data,path):
    '''
    Ecriture de nuage de points sous format ply
    Inputs :
        data = matrice (3 x n)
        path = chemin d'acc�s au fichier
    '''
    write_ply(path, data.T, ['x', 'y', 'z'])

def show_cloud_points(data):
    '''
    Visualisation de nuages de points avec MatplotLib'
    Input :
        data = matrice (3 x n)
    '''
    #plt.cla()
    # Aide en ligne : help(plt)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data[0], data[1], data[2], '.')
    #ax.plot(data_aligned[0], data_aligned[1], data_aligned[2], '.')
    #plt.axis('equal')
    plt.show()


def decimate(points: np.ndarray[float], k: int, method: str = 'for') -> np.ndarray[float]:
    '''
    D�cimation
    # ----------
    Inputs :
        data = matrice (3 x n)
        k_ech : facteur de d�cimation
    Output :
        decimated = matrice (3 x (n/k_ech))
    '''
    decimated = []

    if method == 'for':
        for i, point in enumerate(points):
            if i % k == 0:
                decimated.append(point)
        # # 1�re m�thode : boucle for
        # n = data.shape[1]
        # n_ech=int(n/k_ech)

        # decimated = np.vstack(data[:, 0])
            # compl�ter par une boucle for i in range
            # Xi = vecteur du rang k_ech*i (utiliser np.vstack)
            # concat�ner Xi � decimated en utilisant np.hstack

    #else:
        # 2e m�thode : fonction de Numpy array
        #decimated = sous-tableau des colonnes espac�es de k_ech

    return np.array(decimated)




def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of point and "d" the dimension
         ref = (d x N) matrix where "N" is the number of point and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    # TODO
    # Barycenters
    # d�finir les baycentres ref_center et data_center

    # Centered clouds
    # calculer les nuages de points centr�s ref_c et data_c

    # H matrix
    # calculer la matrice H

    # SVD on H
    # calculer U, S, et Vt en utilisant np.linalg.svd

    # Checking R determinant
    # si le d�terminant de U est -1, prendre son oppos�

    # Getting R and T
    # calculer R et T

    return R, T


def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iteratice closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N) matrix where "N" is the number of point and "d" the dimension
        ref = (d x N) matrix where "N" is the number of point and "d" the dimension
        max_iter = stop condition on the number of iteration
        RMS_threshold = stop condition on the distance
    Returns :
        R = (d x d) rotation matrix aligning data on ref
        T = (d x 1) translation vector aligning data on ref
        data_aligned = data aligned on ref
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Create a neighbor structure on ref
    search_tree = KDTree(ref.T)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    for i in range(max_iter):

        # Find the nearest neighbors
        distances, indices = search_tree.query(data_aligned.T, return_distance=True)

        # Compute average distance
        RMS = np.sqrt(np.mean(np.power(distances, 2)))

        # Distance criteria
        if RMS < RMS_threshold:
            break

        # Find best transform
        R, T = best_rigid_transform(data, ref[:, indices.ravel()])

        # Update lists
        R_list.append(R)
        T_list.append(T)
        neighbors_list.append(indices.ravel())
        RMS_list.append(RMS)

        # Aligned data
        data_aligned = R.dot(data) + T


    return data_aligned, R_list, T_list, neighbors_list, RMS_list





#
#           Main
#       \**********/
#

if __name__ == '__main__':


    # Fichiers de nuages de points
    bunny_o_path = 'data/bunny_original.ply'
    bunny_p_path = 'data/bunny_perturbed.ply'
    bunny_r_path = 'data/bunny_returned.ply'
    NDC_o_path = 'data/Notre_Dame_Des_Champs_1.ply'
    NDC_r_path = 'data/Notre_Dame_Des_Champs_returned.ply'

    # Lecture des fichiers
    bunny_o=read_data_ply(bunny_o_path)                    
    bunny_p=read_data_ply(bunny_p_path)
    NDC_o=read_data_ply(NDC_o_path)

    # TODO
    # Visualisation du fichier d'origine
    if True:
        show_cloud_points(bunny_o)

    # Transformations : d�cimation, rotation, translation, �chelle
    # ------------------------------------------------------------
    if True:
        # D�cimation
        k_ech=10
        decimated = decimate(bunny_o,k_ech)

        # Visualisation sous Python et par �criture de fichier
        show_cloud_points(decimated)

        # Visualisation sous CloudCompare apr�s �criture de fichier
        write_data_ply(decimated,bunny_r_path)
        # Puis ouvrir le fichier sous CloudCompare pour le visualiser

    if False:
        show_cloud_points(NDC_o)
        decimated = decimate(NDC_o,1000)
        show_cloud_points(decimated)
        write_data_ply(decimated,NDC_r_path)

    if False:        
        # Translation
        # translation = d�finir vecteur [0, -0.1, 0.1] avec np.array et reshape
        points=bunny_o + translation
        show_cloud_points(points)
        
        # Find the centroid of the cloud and center it
        #centroid = barycentre - utiliser np.mean(points, axis=1) et reshape
        points = points - centroid
        show_cloud_points(points)
        
        # Echelle
        # points = points divis�s par 2
        show_cloud_points(points)
        
        # Define the rotation matrix (rotation of angle around z-axis)
        # angle de pi/3,
        # d�finir R avec np.array et les cos et sin.
        
        # Apply the rotation
        points=bunny_o
        # centrer le nuage de points        
        # appliquer la rotation - utiliser la fonction .dot
        # appliquer la translation oppos�e
        show_cloud_points(points)


    # Meilleure transformation rigide (R,Tr) entre nuages de points
    # -------------------------------------------------------------
    if False:

        show_cloud_points(bunny_p)
        
        # Find the best transformation
        R, Tr = best_rigid_transform(bunny_p, bunny_o)
        
        
        # Apply the tranformation
        opt = R.dot(bunny_p) + Tr
        bunny_r_opt = opt
        
        # Show and save cloud
        show_cloud_points(bunny_r_opt)
        write_data_ply(bunny_r_opt,bunny_r_path)
        
        
        # Get average distances
        distances2_before = np.sum(np.power(bunny_p - bunny_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(bunny_r_opt - bunny_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))
        
        print('Average RMS between points :')
        print('Before = {:.3f}'.format(RMS_before))
        print(' After = {:.3f}'.format(RMS_after))
    
   
    # Test ICP and visualize
    # **********************
    if False:

        bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(bunny_p, bunny_o, 25, 1e-4)
        plt.plot(RMS_list)
        plt.show()





