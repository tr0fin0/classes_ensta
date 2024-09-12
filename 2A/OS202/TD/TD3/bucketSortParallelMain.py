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
size = 1e7



# chaque processus avec les donnes
# were one process creates the entire array of random numbers but this is not efficient. each process can creates a slice of the data and pass to the respectivy bucket with the same index function used previously.
# even with it is costly to declare and read the array it will be faster with each process can 
if process == root:
    # main process:
    # generate data
    np.random.seed = 1
    array = np.random.rand(int(size))

    start = time.time()
    buckets = []
    sortedArray = []


    # create buckets
    for i in range(nProcess-1):
        buckets.append([])


    # adding values
    for value in array:
        index = int(value * (nProcess-1))
        buckets[index].append(value)


    for i in range(nProcess-1):
        # there is one bucket for each process
        comm.send(buckets[i], dest = (i+1))

    for i in range(nProcess-1):
        sortedBucket = comm.recv(source = (i+1))

        sortedArray += sortedBucket # array concatenate

    end = time.time()
    print(f'[{(end - start):2.6f} s]: parallelMain ({size})')


else:
    bucket = comm.recv(source = root)
    comm.send(sorted(bucket), dest=root)


# benchmark:
#   [27.253314 s]: parallel (10000000.0) | original code
#   [33.639024 s]: parallel (10000000.0) | w/ concatenate arrays
#   [20.876480 s]: parallel (10000000.0)
#   [22.114769 s]: parallel (10000000.0) | remove sizeBuckets