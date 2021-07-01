import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt

# X = [u, v]
a = 1.
b = 0.1
c = 1.5
d = 0.75
e = 0.1

def dX1_dt(X, t):
    u, v = X
    return np.array([a * u - b * u * v, -c * v + d * b * u * v])

def dX2_dt(X, t):
    u, v = X
    return np.array([a * u - b * u * v - e * u ** 2, -c * v + d * b * u * v])

def Model(dX_dt):
    t = np.linspace(0, 15, 1000)
    X0 = np.array([10, 5])
    X = odeint(dX_dt, X0, t)

    X = np.transpose(X)
    rabbits = X[0]
    foxes = X[1]

    plt.plot(t, rabbits, "r-", label='Rabbits')
    plt.plot(t, foxes, "b-", label='Foxes')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.show()
if __name__ == '__main__':
    Model(dX1_dt)
    Model(dX2_dt) #В этой модифицированной модели,
                  #популяция лисов при увеличении времени
                  #достигнет нуля, и популяция зайцев
                  #всегда будет равна какой-нибудь константе


