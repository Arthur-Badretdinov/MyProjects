import numpy as np
from scipy import linalg as lin
from matplotlib import pyplot as plt
import time

def odu4_1(koeffs, func, L, phi, psi, N):
    a4, a3, a2, a1, a0 = koeffs
    lphi, rphi = phi
    lpsi, rpsi = psi
    h = L / N

    Ad2 = np.ones(N - 1) #y[n-2]
    Ad1 = np.ones(N) #y[n-1]
    Al = np.ones(N + 1) #y[n]
    Au1 = np.ones(N) #y[n+1]
    Au2 = np.ones(N - 1) #y[n+2]

    Ad2[::] = a4 / h ** 4 - a3 / (2 * h ** 3)
    Ad1[::] = -4 * a4 / h ** 4 + a3 / h ** 3 + a2 / h ** 2 - a1 / (2 * h)
    Al[::] = 6 * a4 / h ** 4 - 2 * a2 / h ** 2 + a0
    Au1[::] = -4 * a4 / h ** 4 - a3 / h ** 3 + a2 / h ** 2 + a1 / (2 * h)
    Au2[::] = a4 / h ** 4 + a3 / (2 * h ** 3)
    F = np.fromfunction(func, (N + 1, 1))

    Al[0] = 1
    Au1[0] = 0
    Au2[0] = 0
    F[0] = lphi

    Ad1[0] = -1 / h
    Al[1] = 1 / h
    Au1[1] = 0
    Au2[1] = 0
    F[1] = lpsi

    Ad2[N - 3] = 0
    Ad1[N - 2] = 0
    Al[N - 1] = 1 / h
    Au1[N - 1] = -1 / h
    F[N - 1] = rpsi

    Ad2[N - 2] = 0
    Ad1[N - 1] = 0
    Al[N] = 1
    F[N] = rphi

    A = np.diag(Ad2, -2) + np.diag(Ad1, -1) + np.diag(Al) + np.diag(Au1, 1) + np.diag(Au2, 2)
    res = lin.solve(A, F)
    return res


def odu4_2(koeffs, func, L, phi, psi, N):
    a4, a3, a2, a1, a0 = koeffs
    lphi, rphi = phi
    lpsi, rpsi = psi
    h = L / N

    Ad2 = np.ones(N + 1)  # y[n-2]
    Ad1 = np.ones(N + 1)  # y[n-1]
    Al = np.ones(N + 1)  # y[n]
    Au1 = np.ones(N + 1)  # y[n+1]
    Au2 = np.ones(N + 1)  # y[n+2]

    Ad2[::] = a4 / h ** 4 - a3 / (2 * h ** 3)
    Ad1[::] = -4 * a4 / h ** 4 + a3 / h ** 3 + a2 / h ** 2 - a1 / (2 * h)
    Al[::] = 6 * a4 / h ** 4 - 2 * a2 / h ** 2 + a0
    Au1[::] = -4 * a4 / h ** 4 - a3 / h ** 3 + a2 / h ** 2 + a1 / (2 * h)
    Au2[::] = a4 / h ** 4 + a3 / (2 * h ** 3)
    F = np.fromfunction(func, (N + 1, 1))

    Ad2[0] = 0
    Ad1[0] = 0
    Al[0] = 1
    Au1[0] = 0
    Au2[0] = 0
    F[0] = lphi

    Ad2[1] = 0
    Ad1[1] = -1 / h
    Al[1] = 1 / h
    Au1[1] = 0
    Au2[1] = 0
    F[1] = lpsi

    Ad2[N - 1] = 0
    Ad1[N - 1] = 0
    Al[N - 1] = 1 / h
    Au1[N - 1] = -1 / h
    Au2[N - 1] = 0
    F[N - 1] = rpsi

    Ad2[N] = 0
    Ad1[N] = 0
    Al[N] = 1
    Au1[N] = 0
    Au2[N] = 0
    F[N] = rphi

    Ad2 = np.roll(Ad2, -2)
    Ad1 = np.roll(Ad1, -1)
    Au1 = np.roll(Au1, 1)
    Au2 = np.roll(Au2, 2)

    A_band = np.concatenate((Au2, Au1, Al, Ad1, Ad2)).reshape(5, N + 1)

    res = lin.solve_banded((2, 2), A_band, F)
    return res

def solve_and_plot(L, N):
    print("N =", N)
    start_time = time.time()
    res1 = odu4_1([1, -1, 1, -1, 1], lambda x, y: -x * L / N, L, (1, 0), (0, -1), N)
    print("Time for lin.solve:", time.time() - start_time)
    start_time = time.time()
    res2 = odu4_2([1, -1, 1, -1, 1], lambda x, y: -x * L / N, L, (1, 0), (0, -1), N)
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


