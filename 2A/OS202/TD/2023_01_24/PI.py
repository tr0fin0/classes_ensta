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



def plotEstimation(input, title: str) -> None:
    # let us make a simple graph
    fig = plt.figure(figsize=[8, 8])
    ax  = plt.subplot(111)

    # set the basic properties
    ax.set_xlabel('n')
    ax.set_ylabel('pi')
    ax.set_title(title)

    ax.hlines(y=PI, xmin=0, xmax=(len(input)-1), linewidth=1, linestyles='--')

    # set the grid on
    ax.grid('on')

    plt.plot(input)
    plt.show()

    return None
def monteCarloPi(n: int) -> float:
    """
    https://www.youtube.com/watch?v=prPyPvjvfqM
    """
    points = np.random.uniform(-1, 1, (n,2))
    inside = np.sum(points[:,0]**2 + points[:,1]**2 <= 1)

    if n == 0: return 0

    return 4 * inside / n
MAX_POINTS = 1e4

def estimatePi(points: int, showPlot: bool = True) -> None:
    estimation = []

    for n in range(points):
        estimation.append(monteCarloPi(n))

    pi_mean = np.mean(estimation)
    error   = pi_mean - PI


    print(f'pi: {pi_mean:1.6f} error: {error:1.6f}')

    if showPlot == True:
        plotEstimation(estimation, "stochastique")

    return None


def estimatePiMultithread(points: int) -> None:
    """
    https://gist.github.com/amitsaha/2036026
    """

    totalCPUs = multiprocessing.cpu_count()
    print(f'CPUs: {totalCPUs:2.0f}')

    # iterable with a list of points to generate in each worker
    # each worker process gets n/np number of points to calculate Pi from

    pointsCPUs = [int(points/totalCPUs) for i in range(totalCPUs)]

    #Create the worker pool
    # http://docs.python.org/library/multiprocessing.html#module-multiprocessing.pool
    pool = Pool(processes=totalCPUs)

    # parallel map
    estimation  = pool.map(monteCarloPi, pointsCPUs)

    pi_mean = np.mean(estimation)
    error   = pi_mean - PI


    print(f'pi: {pi_mean:1.6f} error: {error:1.6f}')

    return None


PI = np.pi
MIN_POINTS = 0
MAX_POINTS = 1e8

def main() -> None:

    # start = time.time()
    # estimatePi(int(MAX_POINTS), showPlot=False)
    # end = time.time()
    # print(f'serial: {(end - start):2.6f} s')


    start = time.time()
    estimatePiMultithread(int(MAX_POINTS))
    end = time.time()
    print(f'parrel: {(end - start):2.6f} s')

    return None



main()