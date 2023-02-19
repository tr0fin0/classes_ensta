#!/usr/bin/env python3
import numpy as np
import time
from mpi4py import MPI




def main():
    # initialization MPI
    # mpirun -np 10 python3 bucketSort.py
    comm = MPI.COMM_WORLD
    nProcess = comm.Get_size()
    process  = comm.Get_rank()
    root = 0

    def getRandomArray(nProcess: int = 25, seed:int = 1) -> list:
        np.random.seed = seed

        return np.random.rand(nProcess)


    def bucketSortParallel(arraySize: float = 1e7) -> list:

        if process == root:
            # main process:
            # generate data
            array = getRandomArray(int(arraySize))
            buckets = []

        if process == root:
            # create buckets
            for i in range(nProcess):
            # for i in range(nProcess - 1):
                buckets.append([])

            # adding values
            for value in array:
                index = int(value * (nProcess))
                # index = int(value * (nProcess - 1))
                buckets[index].append(value)


            # sendcounts = np.array(comm.gather(len(sendbuf), root))
            # for i in range(1, (nProcess+1)):
            #     comm.send(buckets[i], dest=(i))
            #     print(f'0: {i}')

            # sortedArray = []
            # recvbuf = np.empty(sum(sendcounts), dtype=float)
        else:
            buckets = None
            # buckets = comm.recv(source=0)
            # recvbuf = None
            # sendbuf = np.array(buckets)

        # if process == 0:
        #     sortedArray = comm.gather(buckets, root=0)
        # else:
        #     buckets = sorted(buckets)

        # # each process receives a bucket and sort it
        buckets = sorted(comm.scatter(buckets, 0))

        # # each process sends it's sorted bucket to main process
        sortedArray = comm.gather(buckets, root=root)


        # comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=root)

        return sortedArray


    # bucket sort
    start = time.time()
    arraySorted = bucketSortParallel()
    end = time.time()

    # TODO code runs multiple times

    if process == 0:
        print(f'[{(end - start):2.6f} s]: parallel')
        # print(f'{arraySorted}')




if __name__ == "__main__":
    main()