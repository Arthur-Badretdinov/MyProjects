import numpy as np
from scipy.optimize import leastsq
from matplotlib import pyplot as plt

def Task2_2_1():
    T = np.array([20, 22, 24, 26, 28])
    R = np.array([40, 38.4, 36.9, 35.5, 34.1])

    def err_func(p, T, R):
        return R - p[0] * T - p[1]

    p = np.zeros(2)
    p, tmp = leastsq(err_func, p, args=(T, R))

    print('R =', p[0], '* T +', p[1])
    print('T = 21: R =', p[0] * 21 + p[1])
    for i in range(5):
        print('Невязка при i =', i + 1, ':', abs(R[i] - p[0] * T[i] - p[1]))

    plt.plot(T, R, "ro")
    plt.plot(T, p[0] * T + p[1])
    plt.show()

def Task2_2_2():
    T = np.array([0, 0.083008, 0.166016, 0.249024, 0.332032, 0.41504])
    N = np.array([1000725, 942638, 892394, 840251, 795101, 749042])
    N0 = N[0]

    def err_func(l, T, N):
        return np.log(N) + l * T - np.log(N0)

    l = 0
    l, tmp = leastsq(err_func, l, args=(T,N))
    print('lambda:', l[0])
    print('T1/2:', np.log(2) / l[0])
    print('Сумма квадратов невязок:', sum([(np.log(N[i]) - np.log(N0) + l[0] * T[i]) ** 2 for i in range(5)]))

    plt.plot(T, N, "ro")
    plt.plot(T, N0 * np.exp(-l[0] * T))
    plt.show()

if __name__ == '__main__':
    #Task2_2_1()
    Task2_2_2()