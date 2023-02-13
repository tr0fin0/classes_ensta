import numpy as np
import time




def getRandomArray(nProcess: int = 25, seed:int = 1) -> list:
    np.random.seed = seed

    return np.random.rand(nProcess)


def printArrayInfo(array: list) -> None:
    print(f'min:  {min(array):1.6f}')
    print(f'mean: {np.mean(array):1.6f}')
    print(f'max:  {max(array):1.6f}')

    printArray(array)
    return None


def printArray(array: list) -> None:
    for index in range(len(array)):
        print(f'{index:3.0f}: {array[index]:1.6f}')

    return None


def bucketSort(array: list, numBuckets: int = 10) -> list:
    buckets = []

    # create buckets
    for i in range(numBuckets):
        buckets.append([])

    # adding values to 
    for value in array:
        index = int(value * numBuckets)
        buckets[index].append(value)

    for index in range(len(buckets)):
        buckets[index] = sorted(buckets[index])

    sortedArray = []
    for bucket in buckets:
        for value in bucket:
            sortedArray.append(value)

    return sortedArray

def execution() -> None:
    # initialization random array
    array = getRandomArray(int(1e7))
    # printArrayInfo(array)   # debug


    # bucket sort
    start = time.time()
    arraySorted = bucketSort(array)
    end = time.time()
    print(f'[{(end - start):2.6f} s]: serial')


    # verification
    # start = time.time()
    # if (arraySorted == sorted(array)):
    #     end = time.time()
    #     print(f"[{(end - start):2.6f} s]: sorting successful")

    return None





def main():
    execution()
    # print([0]*10)




if __name__ == "__main__":
    main()