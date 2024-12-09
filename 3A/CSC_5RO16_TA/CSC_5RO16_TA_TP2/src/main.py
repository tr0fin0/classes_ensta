"""
RRT_2D
@author: huiming zhou

Modified by David Filliat
"""

import os
import sys
import math
import numpy as np
import plotting, utils
import env
import rrt
import rrt_star

# parameters
showAnimation = True

def compute_rrt():
    x_start=(2, 2)  # Starting node
    x_goal=(49, 24)  # Goal node
    environment = env.Env()

    rrt_object = rrt.Rrt(environment, x_start, x_goal, 2, 0.10, 1500)
    path, nb_iter = rrt_object.planning()

    if path:
        print('Found path in ' + str(nb_iter) + ' iterations, length : ' + str(rrt.get_path_length(path)))
        if showAnimation:
            rrt_object.plotting.animation(rrt_object.vertex, path, "RRT", True)
            plotting.plt.show()
    else:
        print("No Path Found in " + str(nb_iter) + " iterations!")
        if showAnimation:
            rrt_object.plotting.animation(rrt_object.vertex, [], "RRT", True)
            plotting.plt.show()


def compute_rrt_star():
    x_start = (2, 2)  # Starting node
    x_goal = (49, 24)  # Goal node
    environment = env.Env()

    rrt_star_object = rrt_star.RrtStar(environment, x_start, x_goal, 2, 0.10, 20, 1500)
    path, nb_iter = rrt_star_object.planning()

    if path:
        print('Found path in ' + str(nb_iter) + ' iterations, length : ' + str(rrt_star.get_path_length(path)))
        if showAnimation:
            rrt_star_object.plotting.animation(rrt_star_object.vertex, path, "RRT*", True)
            plotting.plt.savefig('rrt_star.png')
            plotting.plt.show()
    else:
        print("No Path Found in " + str(nb_iter) + " iterations!")
        if showAnimation:
            rrt_star_object.plotting.animation(rrt_star_object.vertex, [], "RRT*", True)
            plotting.plt.savefig('rrt_star.png')
            plotting.plt.show()


def main():
    # compute_rrt()
    compute_rrt_star()


if __name__ == '__main__':
    main()