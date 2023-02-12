import numpy as np




def getRandomArray(size: int = 100, seed:int = 1) -> list:
    np.random.seed = seed

    return np.random.rand(size)


def printArrayInfo(array: list) -> None:
    print(f'min:  {min(array):1.6f}')
    print(f'mean: {np.mean(array):1.6f}')
    print(f'max:  {max(array):1.6f}')

    return None




def main():

    array = getRandomArray()
    printArrayInfo(array)




if __name__ == "__main__":
    main()