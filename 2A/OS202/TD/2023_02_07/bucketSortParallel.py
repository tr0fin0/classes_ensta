import numpy as np
import time
from mpi4py import MPI




def main():
    # initialization MPI
    # mpirun -np 4 python3 bucketSort.py
    comm = MPI.COMM_WORLD
    nProcess = comm.Get_size()
    process  = comm.Get_rank()

    def getRandomArray(nProcess: int = 25, seed:int = 1) -> list:
        np.random.seed = seed

        return np.random.rand(nProcess)


    def bucketSortParallel(arraySize: float = 1e1) -> list:
        if process == 0:
            # main process:
            # generate data
            array = getRandomArray(int(arraySize))
            buckets = []

            # create buckets
            for i in range(nProcess):
                buckets.append([])

            # adding values
            for value in array:
                index = int(value * nProcess)
                buckets[index].append(value)
        else:
            buckets = None

        # each process receives a bucket and sort it
        buckets = sorted(comm.scatter(buckets, 0))

        # each process sends it's sorted bucket to main process
        sortedArray = comm.gather(buckets, root=0)

        return sortedArray


    # bucket sort
    start = time.time()
    arraySorted = bucketSortParallel()
    end = time.time()

    # TODO code runs multiple times

    if process == 0:
        print(f'[{(end - start):2.6f} s]: parallel')
        print(f'{arraySorted}')




if __name__ == "__main__":
    main()