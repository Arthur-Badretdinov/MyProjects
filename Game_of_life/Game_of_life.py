import numpy as np
from matplotlib import pyplot as plt

def matrix_generation(M, N):
    return np.random.randint(0, 2, (M, N))

def iterate_numpy(M):
    neighbors = sum([np.roll(np.roll(M, -1, 1), 1, 0),
                     np.roll(np.roll(M, 1, 1), -1, 0),
                     np.roll(np.roll(M, 1, 1), 1, 0),
                     np.roll(np.roll(M, -1, 1), -1, 0),
                     np.roll(M, 1, 1),
                     np.roll(M, -1, 1),
                     np.roll(M, 1, 0),
                     np.roll(M, -1, 0)])
    return (neighbors == 3) | (M & (neighbors == 2))

if __name__ == '__main__':
    Matrix = matrix_generation(512, 256)
    for i in range(int(input())):
        Matrix = iterate_numpy(Matrix)
    plt.matshow(Matrix)
    plt.show()
