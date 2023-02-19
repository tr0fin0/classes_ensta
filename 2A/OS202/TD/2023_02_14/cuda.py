import numpy as np
import matplotlib.pylab as plt

from numba import vectorize
from numba import cuda, jit
from timeit import default_timer as timer
from matplotlib.pylab import imshow, show

# this code was developed based on the following video:
#   https://www.youtube.com/watch?v=-lcWV4wkHsk



N = int(64 * 1e4)
A = np.ones(N, dtype = np.float32)
B = np.ones(N, dtype = np.float32)
C = np.ones(N, dtype = np.float32)


def CPU() -> None:
    print('CPU____')

    # ================================================
    def vectorMultipliy(a, b, c):
        for i in range(a.size):
            c[i] = a[i] * b[i]

    start = timer()
    vectorMultipliy(A, B, C)
    totalTime = timer() - start

    print(f'\tvectorMultipliy:  {totalTime:2.4f} s')


    # ================================================
    def fillArray(a):
        for i in range(a.size):
            a[i] += 1

    start = timer()
    fillArray(A)
    totalTime = timer() - start

    print(f'\tfillArray:        {totalTime:2.4f} s')


    # ================================================
    def mandelbrot(x, y, interations):
        c = complex(x, y)
        z = 0.0j
        i = 0

        for i in range(interations):
            z = z**2 + c

            if (z.real * z.real + z.imag * z.imag) >= 4:
                return i

            return 255


    def createFractal(xMin, xMax, yMin, yMax, image, interations):
        width  = image.shape[1]
        height = image.shape[0]

        pixel_size_x = (xMax - xMin)/width
        pixel_size_y = (yMax - yMin)/height

        for x in range(width):
            real = xMin + x * pixel_size_x

            for y in range(height):
                imag = yMin + y * pixel_size_y
                color= mandelbrot(real, imag, interations)

                image[y, x]= color

    image = np.zeros((500*10, 750*10), dtype = np.uint8)

    start = timer()
    xMin, xMax, yMin, yMax = -2.0, +1.0, -1.0, +1.0
    createFractal(-2.0, +1.0, -1.0, +1.0, image, 20)
    totalTime = timer() - start


    print(f'\tmandelbrot:       {totalTime:2.4f} s')
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.xlabel("real")
    plt.ylabel("imaginary")
    # plt.xlim(xMin, xMax)
    # plt.ylim(yMin, yMax)
    plt.savefig("plot.png")
    plt.imshow(image)

            

    return None


def GPU() -> None:
    print('GPU____')


    # ================================================
    @vectorize(["float32(float32, float32)"], target = 'cuda')
    def vectorMultipliyGPU(a, b):
        return a * b

    start = timer()
    C = vectorMultipliyGPU(A, B)
    totalTime = timer() - start

    print(f'\tvectorMultipliy:  {totalTime:2.4f} s')


    # ================================================
    # just in time compilation
    @jit(target_backend='cuda')
    def fillArrayGPU(a):
        for i in range(a.size):
            a[i] += 1

    start = timer()
    fillArrayGPU(A)
    totalTime = timer() - start

    print(f'\tfillArray:        {totalTime:2.4f} s')


    # ================================================
    @cuda.jit(device=True)
    def mandelbrot(x, y, interations):
        c = complex(x, y)
        z = 0.0j
        i = 0

        for i in range(interations):
            z = z**2 + c

            if (z.real * z.real + z.imag * z.imag) >= 4:
                return i

            return 255


    @cuda.jit
    def createFractal(xMin, xMax, yMin, yMax, image, interations):
        width  = image.shape[1]
        height = image.shape[0]

        pixel_size_x = (xMax - xMin)/width
        pixel_size_y = (yMax - yMin)/height

        x, y = cuda.grid(2)
        if x < width and y < height:
            real = xMin + x * pixel_size_x
            imag = yMin + y * pixel_size_y
            color= mandelbrot(real, imag, interations)
            image[y, x] = color

    ySize = 500*10*2
    xSize = 750*10*2
    image = np.zeros((ySize, xSize), dtype = np.uint8)
    
    pixels = ySize * xSize
    nthreads = 32
    nblocksy = (ySize//nthreads) + 1
    nblocksx = (xSize//nthreads) + 1

    start = timer()
    createFractal[(nblocksx, nblocksy), (nthreads, nthreads)](
        -2.0, +1.0, -1.0, +1.0, image, 100
    )
    totalTime = timer() - start

    print(f'\tmandelbrot:       {totalTime:2.4f} s')
    if False:
        plt.rcParams['figure.figsize'] = [10, 10]
        plt.xlabel("real")
        plt.ylabel("imaginary")
        # plt.xlim(-xSize, xSize)
        # plt.ylim(-ySize, ySize)
        plt.savefig("plot.png")
        plt.imshow(image)
        
    return None




def main():
    # CPU()
    print()
    GPU()


main()