import numpy as np

def approximate_pi(n):
    """
    https://www.youtube.com/watch?v=prPyPvjvfqM
    """
    points = np.random.uniform(-1, 1, (n,2))
    inside = np.sum(points[:,0]**2 + points[:,1]**2 <= 1)

    return 4 * inside / n




MIN_POINTS = 1000
MAX_POINTS = 1000000

def main():
    estimation = []
    points = range(MIN_POINTS, MAX_POINTS, 1)

    for n in points:
        estimation.append(approximate_pi(n))

    
