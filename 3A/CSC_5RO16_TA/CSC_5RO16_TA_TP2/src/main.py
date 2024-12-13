"""
CSC_5RO16_TA_TP2, RRT 2D.

@author: Guilherm TROFINO

Based on code by: David Filliat and Huiming Zhou.
"""

import os
import sys
import math
import numpy as np
import plotting, utils
import env
import rrt
import rrt_star



def compute_rrt(
        environment: env,
        start: tuple[float] = (2, 2),
        goal: tuple[float] = (49, 24),
        step: int = 2,
        goal_sample_rate: float = 0.10,
        max_iterations: int = 1500,
        show_animation: bool = False,
    ) -> None:
    """
    Execute Rapidly Exploring Random Trees alogorithm.

    Args:
        environment (env) : simulation environment.
        start (tuple[float]) : start coordinates. Default value is '(2, 2)'.
        goal (tuple[float]) : goal coordinates. Default value is '(49, 24)'.
        step (int) : execution step size. Default value is '2'.
        goal_sample_rate (float) : percetage of goal sample rate. Default value is '0.10'.
        max_iterations (int) : iterations limit. Default value is '1500'.
        show_animation (bool) : show execution animation? Default value is 'False'.
    """
    rrt_object = rrt.Rrt(environment, start, goal, step, goal_sample_rate, max_iterations)
    path, nb_iter = rrt_object.planning()

    print(f'RRT, {step:4d}, {goal_sample_rate:1.4f}, {max_iterations:4d}, {nb_iter:4d}, {rrt.get_path_length(path) if path else 00.0000:4.4f}')

    if show_animation:
        rrt_object.plotting.animation(rrt_object.vertex, path if path else [], "RRT", True)

        plotting.plt.savefig('rrt.png')
        plotting.plt.show()

    plotting.plt.close()


def compute_rrt_star(
        environment: env,
        start: tuple[float] = (2, 2),
        goal: tuple[float] = (49, 24),
        step: int = 2,
        goal_sample_rate: float = 0.10,
        max_iterations: int = 1500,
        show_animation: bool = False,
    ) -> None:
    """
    Execute Rapidly Exploring Random Trees alogorithm modified.

    Args:
        environment (env) : simulation environment.
        start (tuple[float]) : start coordinates. Default value is '(2, 2)'.
        goal (tuple[float]) : goal coordinates. Default value is '(49, 24)'.
        step (int) : execution step size. Default value is '2'.
        goal_sample_rate (float) : percetage of goal sample rate. Default value is '0.10'.
        max_iterations (int) : iterations limit. Default value is '1500'.
        show_animation (bool) : show execution animation? Default value is 'False'.
    """

    rrt_star_object = rrt_star.RrtStar(environment, start, goal, step, goal_sample_rate, 20, max_iterations)
    path, nb_iter = rrt_star_object.planning()

    print(f'RRT*, {step:4d}, {goal_sample_rate:1.4f}, {max_iterations:4d}, {nb_iter:4d}, {rrt.get_path_length(path) if path else 00.0000:4.4f}')
    if show_animation:
        rrt_star_object.plotting.animation(rrt_star_object.vertex, path if path else [], "RRT*", True)

        plotting.plt.savefig('rrt_star.png')
        plotting.plt.show()

    plotting.plt.close()



def question_1(show: bool = False):
    environment = env.Env()

    for max_iteration in [375, 750, 1500, 2250, 3000]:
        for _ in range(10):
            compute_rrt(environment, max_iterations=max_iteration, show_animation=show)
            compute_rrt_star(environment, max_iterations=max_iteration, show_animation=show)


def main():
    
    question_1()


if __name__ == '__main__':
    main()
