from scipy import optimize as opt
import numpy as np

def f1(nums):
    x = nums
    f = np.zeros(1)
    f[0] = 2 ** x + 5 * x - 3
    return f

def f2(nums):
    x, y = nums
    f = np.zeros(2)
    f[0] = 2 * x + np.cos(y)
    f[1] = np.sin(x + 1) - y
    return f

if __name__ == '__main__':
    sol = opt.root(f1, np.zeros(1), method='krylov')
    print(sol.x)
    sol = opt.root(f2, np.array([2, 1.2]), method='krylov')
    print(sol.x)