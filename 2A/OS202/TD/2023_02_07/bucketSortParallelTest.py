#!/usr/bin/env python3
import numpy as np
import time
from mpi4py import MPI




# initialization MPI
# mpirun -np 10 python3 bucketSort.py
comm = MPI.COMM_WORLD
nProcess = comm.Get_size()
process  = comm.Get_rank()

root = 0
size = 1e2



# chaque processus avec les donnes
# were one process creates the entire array of random numbers but this is not efficient. each process can creates a slice of the data and pass to the respectivy bucket with the same index function used previously.
# even with it is costly to declare and read the array it will be faster with each process can 
if process == root:
    # main process:
    # generate data
    start = time.time()
    sortedArray = []

    for i in range(nProcess-1):
    # for i in range(nProcess):
        sortedBucket = comm.recv(source = (i+1))
        # sortedBucket = comm.recv(source = (i))

        sortedArray += sortedBucket # array concatenate

    end = time.time()
    print(f'[{(end - start):2.6f} s]: parallelTest ({size})')
    print(f'{sortedArray}')


else:
    np.random.seed = 1
    array = np.random.rand(int(size/(nProcess-1)))
    bucket = []
    buckets = []


    # create buckets
    for i in range(nProcess-1):
    # for i in range(nProcess):
        buckets.append([])


    # adding values
    for value in array:
        index = int(value * (nProcess-1))
        # index = int(value * (nProcess))
        buckets[index].append(value)


    for i in range(nProcess-1):
    # for i in range(nProcess):
        # there is one bucket for each process
        if i != process:
            comm.send(buckets[i], dest = (i+1))
        else:
            bucket += buckets[i]
        # comm.send(buckets[i], dest = (i))


    # bucket = comm.recv(source = root)

    for i in range(nProcess-1):
        if i != process:
            arrayReceived = comm.recv(source = (i+1))
            bucket += arrayReceived

    # bucket += comm.recv(source = MPI.ANY_SOURCE)

    comm.send(sorted(bucket), dest=root)


# benchmark:
#   [27.253314 s]: parallel (10000000.0) | original code
#   [33.639024 s]: parallel (10000000.0) | w/ concatenate arrays
#   [20.876480 s]: parallel (10000000.0)
#   [22.114769 s]: parallel (10000000.0) | remove sizeBuckets