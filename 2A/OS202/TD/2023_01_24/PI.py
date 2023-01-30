import time
import numpy as np
import matplotlib.pyplot as plt


import multiprocessing
from multiprocessing import Pool

# from mpi4py import MPI

# comm = MPI.COMM_WORLD   # instantice the communication world
# size = comm.Get_size()  # size of the communication world
# rank = comm.Get_rank()  # process ID ('rank')

# PID = os.getpid()



    return 4 * inside / n

def plotEstimation(input, title: str):
    # let us make a simple graph
    fig = plt.figure(figsize=[8, 8])
    ax  = plt.subplot(111)

    # set the basic properties
    ax.set_xlabel('n')
    ax.set_ylabel('pi')
    ax.set_title(title)

    ax.hlines(y=np.pi, xmin=0, xmax=(len(input)-1), linewidth=1, linestyles='--')

    # set the grid on
    ax.grid('on')

    plt.plot(input)
    plt.show()


MIN_POINTS = 0
MAX_POINTS = 1e4

def main():
    estimation = []

    # start = time.perf_counter()
    for n in range(int(MAX_POINTS)):
        estimation.append(approximate_pi(n))
    # end = time.perf_counter()

    pi_mean = np.mean(estimation)
    error   = pi_mean - np.pi
    # time    = end - start

    print(f'~pi: {pi_mean:1.6f} error: {error:1.6f}')
    plotEstimation(estimation, "stochastique")


main()