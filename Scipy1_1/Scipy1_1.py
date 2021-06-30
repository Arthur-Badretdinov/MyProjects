import numpy as np
from scipy import linalg as lin
import timeit

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

if __name__ == '__main__':
    L = np.pi
    N = 5
    odu2_1([0, 1], lambda x, y: -x * L / N, L, (0, 1, 0), (1, 0, 1), N)
    odu2_2([0, 1], lambda x, y: -x * L / N, L, (0, 1, 0), (1, 0, 1), N)


