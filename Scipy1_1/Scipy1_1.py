import numpy as np
from scipy import linalg as lin
from matplotlib import pyplot as plt
import time

def odu2_1(koeffs, func, L, bcl, bcr, N):
    a, b = koeffs
    lalpha, lbeta, lgamma = bcl
    ralpha, rbeta, rgamma = bcr
    h = L / N

    Ad1 = np.ones(N)
    Al = np.ones(N + 1)
    Au1 = np.ones(N)

    Ad1[::] = 1 / h ** 2 - a / (2 * h)
    Al[::] = b - 2 / h ** 2
    Au1[::] = 1 / h ** 2 + a / (2 * h)
    F = np.fromfunction(func, (N + 1, 1))

    Al[0] = lbeta - lgamma / h
    Au1[0] = lgamma / h
    F[0] = lalpha

    Ad1[N - 1] = -rgamma / h
    Al[N] = rbeta + rgamma / h
    F[N] = ralpha

    A = np.diag(Ad1, -1) + np.diag(Al) + np.diag(Au1, 1)
    res = lin.solve(A, F)
    return res


def odu2_2(koeffs, func, L, bcl, bcr, N):
    a, b = koeffs
    lalpha, lbeta, lgamma = bcl
    ralpha, rbeta, rgamma = bcr
    h = L / N

    Ad1 = np.zeros(N + 1)  # y[i-1]
    Al = np.ones(N + 1)  # y[i]
    Au1 = np.ones(N + 1)  # y[i+1]

    Ad1[::] = 1 / h ** 2 - a / (2 * h)
    Al[::] = b - 2 / h ** 2
    Au1[::] = 1 / h ** 2 + a / (2 * h)
    F = np.fromfunction(func, (N + 1, 1))

    Ad1[0] = 0
    Al[0] = lbeta - lgamma / h
    Au1[0] = lgamma / h
    F[0] = lalpha

    Ad1[N] = -rgamma / h
    Al[N] = rbeta + rgamma / h
    Au1[N] = 0
    F[N] = ralpha

    Ad1 = np.roll(Ad1, -1)
    Au1 = np.roll(Au1, 1)

    A = np.concatenate((Au1, Al, Ad1)).reshape(3, N + 1)

    res = lin.solve_banded((1, 1), A, F)
    return res

def solve_and_plot(L, N):
    print("N =", N)
    start_time = time.time()
    res1 = odu2_1([0, 1], lambda x, y: -x * L / N, L, (0, 1, 0), (1, 0, 1), N)
    print("Time for lin.solve:", time.time() - start_time)
    start_time = time.time()
    res2 = odu2_2([0, 1], lambda x, y: -x * L / N, L, (0, 1, 0), (1, 0, 1), N)
    print("Time for lin.solve_banded:", time.time() - start_time)
    x = np.linspace(0, L, N + 1)
    plt.plot(x, res1, "r-", label='lin.solve')
    plt.plot(x, res2, "b-", label='lin.solve_banded')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.show()

if __name__ == '__main__':
    L = np.pi
    solve_and_plot(L, 100)
    solve_and_plot(L, 1000)
    solve_and_plot(L, 10000)



