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

    sizeBuckets = [0]*len(buckets)


    # adding values
    for value in array:
        index = int(value * (nProcess-1))
        buckets[index].append(value)

        sizeBuckets[index] = sizeBuckets[index] + 1

    for i in range(nProcess-1):
        # there is one bucket for each process
        comm.send(buckets[i], dest = (i+1))

    for i in range(nProcess-1):
        sortedBucket = comm.recv(source = (i+1))

        for j in range(sizeBuckets[i]):
            sortedArray.append(sortedBucket[j])

    end = time.time()
    print(f'[{(end - start):2.6f} s]: parallel ({size})')
    # print(f'{sizeBuckets}')
    # print(f'{sortedArray}')

else:
    bucket = comm.recv(source = root)
    comm.send(sorted(bucket), dest=root)