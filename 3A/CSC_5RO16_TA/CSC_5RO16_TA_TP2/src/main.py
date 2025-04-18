"""
CSC_5RO16_TA_TP2, RRT 2D.

@author: Guilherm TROFINO

Based on code by: Huiming Zhou and David Filliat.
"""

from datetime import datetime

import csv
import matplotlib.pyplot as plt
import os
import pandas as pd
import time

import plotting, utils
import env
import obrrt
import rrt
import rrt_star



FILE_DATA: str = "data.csv"

FOLDER_IMAGES: str = "images"
FOLDER_DATABASE: str = "database"

PATH_FILE: str = os.path.dirname(os.path.abspath(__file__))
PATH_IMAGES_FOLDER: str = os.path.join(PATH_FILE, FOLDER_IMAGES)
PATH_DATABASE_FILE: str = os.path.join(PATH_FILE, FOLDER_DATABASE, FILE_DATA)



def create_database(
        headers: list,
        has_datetime: bool = True,
        database_path: str = PATH_DATABASE_FILE
    ) -> None:
    """
    Creates database as CSV file.

    Args:
        headers (list) : CSV header values.
        has_datetime (bool) : include datetime? Default value is 'True'.
        database_path (str) : path to database file. Default value is 'PATH_DATABASE_FILE'.
    """
    if not os.path.exists(database_path):
        with open(database_path, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(['datetime'] + headers if has_datetime else headers)


def add_data(
        data: list,
        has_datetime: bool = True,
        database_path: str = PATH_DATABASE_FILE
    ) -> None:
    """
    Add data to database.

    Args:
        data (list) : list of values to be added to database.
        has_datetime (bool) : include datetime? Default value is 'True'.
        database_path (str) : path to database file. Default value is 'PATH_DATABASE_FILE'.
    """
    try:
        with open(database_path, 'a') as file:
            current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

            writer = csv.writer(file)
            writer.writerow([current_time] + data if has_datetime else data)

    except Exception as e:
        raise Exception("Error saving data: {}".format(str(e)))




def compute_obrrt(
        environment: env,
        start: tuple[float] = (2, 2),
        goal: tuple[float] = (49, 24),
        step: float = 2,
        goal_sample_rate: float = 0.10,
        corner_sample_rate: float = 0.10,
        max_iterations: int = 1500,
        save_data: bool = True,
        show_animation: bool = False,
    ) -> None:
    """
    Execute Rapidly Exploring Random Trees alogorithm.

    Args:
        environment (env) : simulation environment.
        start (tuple[float]) : start coordinates. Default value is '(2, 2)'.
        goal (tuple[float]) : goal coordinates. Default value is '(49, 24)'.
        step (int) : execution step size. Default value is '2'.
        goal_sample_rate (float) : percetage of goal sample rate. Default value is '0.00'.
        max_iterations (int) : iterations limit. Default value is '1500'.
        show_animation (bool) : show execution animation? Default value is 'False'.
    """
    start_time = time.time()

    obrrt_object = obrrt.ObRrt(environment, start, goal, step, goal_sample_rate, corner_sample_rate, max_iterations, show_animation)
    path, nb_iter = obrrt_object.planning()

    end_time = time.time()

    if save_data:
        add_data([
            end_time - start_time,
            'OBRRT',
            environment.name,
            step,
            goal_sample_rate,
            corner_sample_rate,
            max_iterations,
            nb_iter,
            obrrt.get_path_length(path) if path else 00.0000
        ])

    if show_animation:
        obrrt_object.plotting.animation(obrrt_object.vertex, path if path else [], "OBRRT", True)

        plotting.plt.xlim([0, 50])
        plotting.plt.ylim([0, 30])
        plotting.plt.tight_layout()
        plotting.plt.savefig(f'{PATH_IMAGES_FOLDER}/map_obrrt_{environment.name}_{step}_{goal_sample_rate}_{corner_sample_rate}_{max_iterations}.png')
        plotting.plt.show()

    plotting.plt.close()


def compute_rrt(
        environment: env,
        start: tuple[float] = (2, 2),
        goal: tuple[float] = (49, 24),
        step: float = 2,
        goal_sample_rate: float = 0.10,
        corner_sample_rate: float = 0.10,
        max_iterations: int = 1500,
        save_data: bool = True,
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
    start_time = time.time()

    rrt_object = rrt.Rrt(environment, start, goal, step, goal_sample_rate, max_iterations, show_animation)
    path, nb_iter = rrt_object.planning()

    end_time = time.time()

    if save_data:
        add_data([
            end_time - start_time,
            'RRT',
            environment.name,
            step,
            goal_sample_rate,
            corner_sample_rate,
            max_iterations,
            nb_iter,
            rrt.get_path_length(path) if path else 00.0000
        ])

    if show_animation:
        rrt_object.plotting.animation(rrt_object.vertex, path if path else [], "RRT", True)

        plotting.plt.xlim([0, 50])
        plotting.plt.ylim([0, 30])
        plotting.plt.tight_layout()
        plotting.plt.savefig(f'{PATH_IMAGES_FOLDER}/map_rrt_{environment.name}_{step}_{goal_sample_rate}_{corner_sample_rate}_{max_iterations}.png')
        plotting.plt.show()

    plotting.plt.close()


def compute_rrt_star(
        environment: env,
        start: tuple[float] = (2, 2),
        goal: tuple[float] = (49, 24),
        step: float = 2,
        corner_sample_rate: float = 0.10,
        goal_sample_rate: float = 0.10,
        max_iterations: int = 1500,
        save_data: bool = True,
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
    start_time = time.time()

    rrt_star_object = rrt_star.RrtStar(environment, start, goal, step, goal_sample_rate, 20, max_iterations, show_animation)
    path, nb_iter = rrt_star_object.planning()

    end_time = time.time()

    if save_data:
        add_data([
            end_time - start_time,
            'RRT*',
            environment.name,
            step,
            goal_sample_rate,
            corner_sample_rate,
            max_iterations,
            nb_iter,
            rrt.get_path_length(path) if path else 00.0000
        ])

    if show_animation:
        rrt_star_object.plotting.animation(rrt_star_object.vertex, path if path else [], "RRT*", True)

        plotting.plt.xlim([0, 50])
        plotting.plt.ylim([0, 30])
        plotting.plt.tight_layout()
        plotting.plt.savefig(f'{PATH_IMAGES_FOLDER}/map_rrt_star_{environment.name}_{step}_{goal_sample_rate}_{corner_sample_rate}_{max_iterations}.png')
        plotting.plt.show()

    plotting.plt.close()


def algorithm_performance(
        environment: str,
        step: float,
        goal_sample_rate: float,
        corner_sample_rate: float,
        bar_width: float = 100,
        save: bool = True,
        show: bool = False
    ) -> None:
    """
    Plot algorithm performance, displaying:
        - path length as bars;
        - execution duration as lines;
        - iterations as text;

    Args:
        step (float) : RRT step size.
        goal_sample_rate (float) : RRT goal sample rate.
        bar_width (float, optional): width of the bars. Defaults is 100.
        save (bool, optional): save the plot? Defaults is True.
        show (bool, optional): show the plot? Defaults is False.
    """
    df = pd.read_csv(PATH_DATABASE_FILE)
    df_filtered = (
        df.query(f"environment == '{environment}' & step == {step} & goal_sample_rate == {goal_sample_rate} & corner_sample_rate == {corner_sample_rate}")
        .groupby(["method", "max_iterations"])
        .agg(
            path_length=("path_length", "mean"),
            duration_execution=("duration_execution", "mean"),
            iterations=("iterations", "mean"),
            repetitions=("path_length", "size"),
            repetitions_success=("path_length", lambda x: (x != 0).sum())
        )
        .reset_index()
    )


    max_iterations = df_filtered["max_iterations"].unique()
    methods = df_filtered["method"].unique()

    fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
    ax_right = ax.twinx()


    for i, method in enumerate(methods):
        positions = max_iterations + i * bar_width
        method_data = df_filtered.query(f"method == '{method}'")

        bars = ax.bar(
            positions,
            method_data["path_length"],
            width=bar_width,
            alpha=0.75,
            edgecolor='black',
            label=method,
        )

        ax_right.plot(
            positions,
            method_data["duration_execution"],
            linewidth=2,
            marker='o',
            markersize=6,
            markerfacecolor='black',
            label=method,
        )


        for j, bar in enumerate(bars):
            pos_x = bar.get_x() + bar.get_width() / 2
            height = bar.get_height()
            duration = method_data["duration_execution"].values[j]
            iterations = method_data["iterations"].values[j]
            repetitions_success = method_data["repetitions_success"].values[j]

            ax.text(
                pos_x,
                height + 1,
                f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize = 10,
            )
            ax.text(
                pos_x,
                90,
                f"{iterations:.1f}",
                ha='center',
                va='bottom',
                fontsize = 8,
                fontstyle = 'oblique',
            )
            ax.text(
                pos_x,
                90,
                f"{repetitions_success:02d}",
                ha='center',
                va='top',
                fontsize = 8,
                fontstyle = 'oblique',
            )
            ax_right.text(
                pos_x,
                duration + 1,
                f"{duration:.2f}",
                ha='center',
                va='bottom',
                fontsize = 10,
            )

    ax.axhline(
        y=51.89,
        color="red",
        linestyle="--",
        linewidth=2,
        label="minimum path length"
    )

    repetitions = df_filtered["repetitions"].iloc[0]
    title = f'average-algorithm-performance_{environment}_{step}_{goal_sample_rate}_{corner_sample_rate}_{repetitions}'
    plt.suptitle(title)

    ax_right.set_ylabel('Average Execution Duration [s]')
    ax_right.set_ylim(0, 400)
    ax_right.set_yticks(range(0, 401, 40))
    ax_right.set_yticklabels([str(i) for i in range(0, 401, 40)])

    ax.set_xticks(max_iterations + bar_width / 2)
    ax.set_xticklabels(max_iterations)
    ax.set_xlabel('Maximum Iterations', fontsize=12)

    ax.set_ylabel('Average Path Length [m]')
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.set_yticklabels([str(i) for i in range(0, 101, 10)])

    plt.grid(True, axis='y', linestyle='--', alpha=0.75)
    plt.legend(loc='upper right')
    plt.tight_layout()

    if save:
        plt.savefig(f'{PATH_IMAGES_FOLDER}/{title}.png', dpi=300, bbox_inches='tight')

    if show:
        plt.show()


def obrrt_performance(
        environment: str,
        step: float,
        goal_sample_rate: float,
        max_iterations: float,
        bar_width: float = 300,
        save: bool = True,
        show: bool = False
    ) -> None:
    """
    Plot algorithm performance, displaying:
        - path length as bars;
        - execution duration as lines;
        - iterations as text;

    Args:
        step (float) : RRT step size.
        goal_sample_rate (float) : RRT goal sample rate.
        bar_width (float, optional): width of the bars. Defaults is 100.
        save (bool, optional): save the plot? Defaults is True.
        show (bool, optional): show the plot? Defaults is False.
    """
    df = pd.read_csv(PATH_DATABASE_FILE)
    df_filtered = (
        df.query(f"environment == '{environment}' & step == {step} & goal_sample_rate == {goal_sample_rate} & max_iterations == {max_iterations}")
        .groupby(["method", "corner_sample_rate"])
        .agg(
            path_length=("path_length", "mean"),
            duration_execution=("duration_execution", "mean"),
            iterations=("iterations", "mean"),
            repetitions=("path_length", "size"),
            repetitions_success=("path_length", lambda x: (x != 0).sum())
        )
        .reset_index()
    )


    corner_sample_rates = df_filtered["corner_sample_rate"].unique()
    methods = df_filtered["method"].unique()

    fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
    ax_right = ax.twinx()

    colors = [plt.cm.tab10(2), plt.cm.tab10(0), plt.cm.tab10(1)]

    for i, method in enumerate(methods):
        positions = corner_sample_rates * 10000 + i * bar_width
        method_data = df_filtered.query(f"method == '{method}'")

        bars = ax.bar(
            positions,
            method_data["path_length"],
            width=bar_width,
            alpha=0.75,
            edgecolor='black',
            color=colors[i],
            label=method,
        )

        ax_right.plot(
            positions,
            method_data["duration_execution"],
            linewidth=2,
            color=colors[i],
            marker='o',
            markersize=6,
            markerfacecolor='black',
            markeredgecolor=colors[i],
            label=method,
        )


        for j, bar in enumerate(bars):
            pos_x = bar.get_x() + bar.get_width() / 2
            height = bar.get_height()
            duration = method_data["duration_execution"].values[j]
            iterations = method_data["iterations"].values[j]
            repetitions_success = method_data["repetitions_success"].values[j]

            ax.text(
                pos_x,
                height + 1,
                f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize = 10,
            )
            ax.text(
                pos_x,
                90,
                f"{iterations:.0f}",
                ha='center',
                va='bottom',
                fontsize = 8,
                fontstyle = 'oblique',
            )
            ax.text(
                pos_x,
                90,
                f"{repetitions_success:02d}",
                ha='center',
                va='top',
                fontsize = 8,
                fontstyle = 'oblique',
            )
            ax_right.text(
                pos_x,
                duration,
                f"{duration:.2f}",
                ha='center',
                va='bottom',
                fontsize = 10,
            )


    ax.axhline(
        y=51.89,
        color="red",
        linestyle="--",
        linewidth=2,
        label="minimum path length"
    )

    repetitions = df_filtered["repetitions"].iloc[0]
    title = f'average-obrrt-performance_{environment}_{step}_{goal_sample_rate}_{max_iterations}_{repetitions}'
    plt.suptitle(title)

    ax_right.set_ylabel('Average Execution Duration [s]')
    ax_right.set_ylim(-6, +4)
    ax_right.set_yticks(range(-6,5,1))
    ax_right.set_yticklabels([f'{i:2d}' for i in range(-6, 5, 1)])

    ax.set_xticks(corner_sample_rates * 10000 + bar_width)
    ax.set_xticklabels(f'{i*100:.0f} %' for i in corner_sample_rates)
    ax.set_xlabel('Corner Sample Rate', fontsize=12)

    ax.set_ylabel('Average Path Length [m]')
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.set_yticklabels([str(i) for i in range(0, 101, 10)])

    plt.grid(True, axis='y', linestyle='--', alpha=0.75)
    plt.legend(loc='upper right')
    plt.tight_layout()

    if save:
        plt.savefig(f'{PATH_IMAGES_FOLDER}/{title}.png', dpi=300, bbox_inches='tight')

    if show:
        plt.show()


def simulation(
        repetitions: int,
        environment = env.Env(),
        steps: list = [1, 2, 4, 8, 16],
        max_iterations: list = [375, 750, 1500, 2250, 3000],
        goal_sample_rate: float = 0.1,
        corner_sample_rates: list = [0.0],
        run_obrrt: bool = False,
        run_rrt: bool = True,
        run_rrt_star: bool = True,
        save_data: bool = True,
        show_animation: bool = False
    ):
    for step in steps:
        for max_iteration in max_iterations:
            for corner_sample_rate in corner_sample_rates:
                for _ in range(repetitions):
                    if run_obrrt:
                        compute_obrrt(
                            environment,
                            step=step,
                            max_iterations=max_iteration,
                            goal_sample_rate=goal_sample_rate,
                            corner_sample_rate=corner_sample_rate,
                            save_data=save_data,
                            show_animation=show_animation
                        )
                    if run_rrt:
                        compute_rrt(
                            environment,
                            step=step,
                            max_iterations=max_iteration,
                            goal_sample_rate=goal_sample_rate,
                            corner_sample_rate=corner_sample_rate,
                            save_data=save_data,
                            show_animation=show_animation
                        )
                    if run_rrt_star:
                        compute_rrt_star(
                            environment,
                            step=step,
                            max_iterations=max_iteration,
                            goal_sample_rate=goal_sample_rate,
                            corner_sample_rate=corner_sample_rate,
                            save_data=save_data,
                            show_animation=show_animation
                        )



def main():
    create_database(
        [
            'duration_execution',
            'method',
            'environment',
            'step',
            'goal_sample_rate',
            'corner_sample_rate',
            'max_iterations',
            'iterations',
            'path_length'
        ]
    )


    # question 1
    simulation(repetitions=10)

    for step in [1, 2, 4, 8, 16]:
        algorithm_performance(
            environment='env', step=step, goal_sample_rate=0.1, corner_sample_rate=0.0
        )


    # question 2
    simulation(
        repetitions=1,
        environment=env.Env2(),
        steps=[2],
        max_iterations=[1500],
        run_obrrt=True,
        run_rrt=False,
        run_rrt_star=False,
        save_data=False,
        show_animation=True
    )

    obrrt_performance(
        environment='env2', step=2, max_iterations=1500, goal_sample_rate=0.1
    )




if __name__ == '__main__':
    main()
