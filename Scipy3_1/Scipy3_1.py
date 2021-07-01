import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt

# X = [s, i]
b = 1
y = 0.1

s0 = 99000
i0 = 1000
r0 = 0
sum = s0 + i0 + r0

def dX_dt(X, t):
    s, i = X
    return np.array([-b * s * i / sum, b * s * i / sum - y * i])

if __name__ == '__main__':
    T = 1
    while 1:
        t = np.linspace(0, T, T + 1)
        X0 = np.array([s0, i0])
        X = odeint(dX_dt, X0, t)
        if X[T][1] < 1:
            break
        T+=1

    X = np.transpose(X)
    s = X[0]
    i = X[1]
    r = sum - s - i

    iidx = np.argmax(i)
    imax = np.max(i)
    print("Maximum infected:", int(imax))
    print("Time during maximum infected:", iidx)

    plt.plot(s, "r-", label='Незаболевшие')
    plt.plot(i, "b-", label='Заболевшие')
    plt.plot(r, "g-", label='Вылеченные')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Время')
    plt.ylabel('Кол-во людей')
    plt.show()

