"""CSC_5RO17_TA_TP4 ICP Cloud Points"""

from matplotlib import pyplot as plt
from matplotlib import collections as mc
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree
from ply import write_ply, read_ply

import numpy as np
import os



def read_cloud(path: str) -> np.ndarray[float]:
    """
    Return cloud points from file path.

    Args:
        path (str) : file path.
    """
    data_ply = read_ply(path)

    return np.vstack((data_ply['x'], data_ply['y'], data_ply['z']))


def write_cloud(points: np.ndarray[float], path: str) -> None:
    """
    Save cloud of points as ply file.

    Args:
        points (np.ndarray[float]) : cloud of points.
        path (str) : file path.
    """
    write_ply(path, points.T, ['x', 'y', 'z'])


def show_cloud(
        points: np.ndarray[float], title: str = 'Cloud of Points', save: bool = False
    ) -> None:
    """
    Show cloud of points as matplotlib plot diagram.

    Args:
        points (np.ndarray[float]) : cloud of points.
        title (str) : title of plot. Default is 'Cloud of Points'.
        save (bool) : save plot? Default value is False.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(points[0], points[1], points[2], '.', label=f'{points.shape[1]} points')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # ax.set_xlim(-0.25, +0.25)
    # ax.set_ylim(-0.25, +0.25)
    # ax.set_zlim(-0.25, +0.25)

    ax.set_title(f'{title}')
    ax.legend()

    plt.tight_layout()

    if save:
        file_path = os.path.abspath(os.path.join(os.getcwd(), f'../outputs/{title}.png'))
        plt.savefig(file_path, dpi=300)

    plt.show()




def compute_ICP(data, ref, max_iter, RMS_threshold):
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
        rotation, translation = compute_rigid_transformation(data, ref[:, indices.ravel()])

        # Update lists
        R_list.append(rotation)
        T_list.append(translation)
        neighbors_list.append(indices.ravel())
        RMS_list.append(RMS)

        # Aligned data
        data_aligned = rotation.dot(data) + translation

    return data_aligned, R_list, T_list, neighbors_list, RMS_list


def compute_rigid_transformation(
        points: np.ndarray[float], ref: np.ndarray[float]
    ) -> list[np.ndarray[float], np.ndarray[float]]:
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
    # centroides
    centroid_ref = np.mean(ref, axis=1).reshape((3, 1))
    centroid_points = np.mean(points, axis=1).reshape((3, 1))

    # centering
    ref = ref - centroid_ref
    points = points - centroid_points

    H = points.dot(ref.T)

    U, S, V = np.linalg.svd(H)

    R = V @ U.T
    T = ref - R @ points

    return R, T


def decimate(points: np.ndarray[float], k: int = 10, method: str = 'for') -> np.ndarray[float]:
    """
    Return resampled by k factor cloud of points as np.ndarray with different methods.

    Args:
        points (np.ndarray[float]) : cloud of points.
        k (int) sample factor. Default is 10.
        method (str) : sample method. Default is 'for'.
    """
    decimated_for = []

    match method.upper():
        case 'FOR':
            for i in range(0, points.shape[1], k):
                decimated_for.append(points[:, i])

        case 'NP':
            return points[:, ::k]

        case _:
            return None

    return np.column_stack(decimated_for)


def get_paths(name: str = 'bunny') -> list[str]:
    """
    Return original_cloud, pertubed_cloud and returned cloud of points absolute path as a list of str.

    Args:
        name (str) : cloud of points name. Default value is 'bunny'.
    """
    match name.upper():
        case 'BUNNY':
            original = os.path.abspath(os.path.join(os.getcwd(), '../data/bunny_original.ply'))
            pertubed = os.path.abspath(os.path.join(os.getcwd(), '../data/bunny_perturbed.ply'))
            returned = os.path.abspath(os.path.join(os.getcwd(), '../data/bunny_returned.ply'))


        case 'NOTRE_DAME_DES_CHAMPS_1':
            original = os.path.abspath(os.path.join(os.getcwd(), '../data/Notre_Dame_Des_Champs_1.ply'))
            pertubed = os.path.abspath(os.path.join(os.getcwd(), '../data/Notre_Dame_Des_Champs_1.ply'))
            returned = os.path.abspath(os.path.join(os.getcwd(), '../data/Notre_Dame_Des_Champs_1.ply'))

        case _:
            return None, None, None

    return original, pertubed, returned


def RMS(points: np.ndarray[float], ref: np.ndarray[float]) -> float:
    return np.sqrt(np.mean(np.sum(np.power(points - ref, 2), axis=0)))


def rotation_matrix(theta: float) -> np.ndarray:
    """"""
    return np.array([
        [+np.cos(theta), -np.sin(theta), 0],
        [+np.sin(theta), +np.cos(theta), 0],
        [0, 0, 1],
    ])


def Q1(cloud_name: str, save: bool) -> None:
    """
    Run Q1 algorithm.

    Args:
        cloud_name (str) : cloud of points analyzed.
        save (bool) : save plot? Default value is False.
    """
    original_path, _, _ = get_paths(cloud_name)
    original_cloud = read_cloud(original_path)

    show_cloud(original_cloud, title=f'{cloud_name}_original', save=save)


def Q2(cloud_name: str, save: bool) -> None:
    """
    Run Q2 algorithm.

    Args:
        cloud_name (str) : cloud of points analyzed.
        save (bool) : save plot? Default value is False.
    """
    original_path, _, _ = get_paths(cloud_name)
    original_cloud = read_cloud(original_path)

    for method in ['for', 'np']:
        for k in [10, 1000]:
            decimated_cloud = decimate(original_cloud, k=k, method=method)

            show_cloud(decimated_cloud, title=f'{cloud_name}_decimated_{method}_{k}', save=save)


def Q3(cloud_name: str, save: bool) -> None:
    """
    Run Q3 algorithm.

    Args:
        cloud_name (str) : cloud of points analyzed.
        save (bool) : save plot? Default value is False.
    """
    original_path, _, _ = get_paths(cloud_name)
    original_cloud = read_cloud(original_path)

    translation = np.array([0, -0.1, +0.1]).reshape((3, 1))
    show_cloud(
        original_cloud + translation,
        title=f'{cloud_name}_translation',
        save=save
    )

    centroid = np.mean(original_cloud, axis=1).reshape((3, 1))
    show_cloud(
        original_cloud - centroid,
        title=f'{cloud_name}_centroid',
        save=save
    )

    scale = 2
    show_cloud(
        original_cloud / scale,
        title=f'{cloud_name}_rescale',
        save=save
    )

    theta = np.pi/3
    show_cloud(
        rotation_matrix(theta).dot(original_cloud),
        title=f'{cloud_name}_rotation_{theta}',
        save=save
    )


def Q4(cloud_name: str, save: bool) -> None:
    """
    Run Q4 algorithm.

    Args:
        cloud_name (str) : cloud of points analyzed.
        save (bool) : save plot? Default value is False.
    """
    original_path, pertubed_path, returned_path = get_paths(cloud_name)
    original_cloud = read_cloud(original_path)

    pertubed_cloud = read_cloud(pertubed_path)
    show_cloud(pertubed_cloud, title=f'{cloud_name}_pertubed', save=save)

    # extract transformation via rigid method
    rotation, translation = compute_rigid_transformation(pertubed_cloud, original_cloud)
    returned_cloud = rotation.dot(pertubed_cloud) + translation

    show_cloud(returned_cloud, title=f'{cloud_name}_transformation', save=save)
    write_cloud(returned_cloud, returned_path)

    # print overall results
    print('Average RMS between points :')
    print(f'RMS pertubed = {RMS(pertubed_cloud, original_cloud):2.4f}')
    print(f'RMS returned = {RMS(returned_cloud, original_cloud):2.4f}')


def Q5(cloud_name: str, save: bool) -> None:
    """
    Run Q5 algorithm.

    Args:
        cloud_name (str) : cloud of points analyzed.
        save (bool) : save plot? Default value is False.
    """
    original_path, _, returned_path = get_paths(cloud_name)
    original_cloud = read_cloud(original_path)
    returned_cloud = read_cloud(returned_path)

    # extract transformation via ICP
    transformation, rotations, translations, neighbors, RMSs = compute_ICP(
        returned_cloud, original_cloud, 25, 1e-4
    )

    plt.plot(RMSs)
    plt.show()



def main():
    """
    Main function execution.

    Note: execution should be done in folder this file is stored
    """
    cloud_name = 'BUNNY'
    # cloud_name = 'NOTRE_DAME_DES_CHAMPS_1'

    # execution variables
    save = False
    visualization = False
    decimation = False
    operations = False
    transforms = False
    ICP = False


    # questions execution
    if visualization: Q1(cloud_name, save)
    if decimation: Q2(cloud_name, save)
    if operations: Q3(cloud_name, save)
    if transforms: Q4(cloud_name, save)
    if ICP: Q5(cloud_name, save)



if __name__ == '__main__':
    main()
