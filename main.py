#!/usr/bin/env python
"""
Demonstration of LFSO utility on least-squares and similar problems
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sp_linalg


class Objfun(object):
    def __init__(self, f, gradf, Lf):
        self._objfun = f
        self._grad = gradf
        self._lfso = Lf

        # Optimization history
        self.history_f = []  # f(xk)
        self.history_g = []  # ||gradf(xk)||
        return

    def obj(self, x):
        return self._objfun(x)

    def grad(self, x):
        return self._grad(x)

    def lfso(self, x, R):
        return self._lfso(x, R)

    def save_iterate(self, x):
        fx = self.obj(x)
        self.history_f.append(fx)
        gx = np.linalg.norm(self.grad(x))
        self.history_g.append(gx)
        return

    def get_history(self):
        return np.array(self.history_f), np.array(self.history_g)

    def clear_history(self):
        self.history_f = []
        self.history_g = []
        return


def generate_lls_problem(n, d, noise_level, ill_conditioned=False):
    # Generate a random linear least-squares problem of size n*d
    A = np.random.randn(n, d)
    if ill_conditioned:
        A[:,0] *= 100  # make somewhat ill-conditioned
    b = A @ np.ones((d,)) + noise_level*np.random.randn(n)
    return A, b


def composite_linear_least_squares(n, d, noise_level, p, ill_conditioned=False):
    """
    Generate a test problem of the form

    f(x) = ||Ax-b||_{2}^{2p} for some p>=1

    where Ax=b is an n*d linear system.

    This problem is written in the form f(x) = h(g(x)) where
    - g(x) = ||Ax-b||_{2}^{2}
    - h(t) = t^p
    """
    assert p >= 1, "Need p>=1 for composite problem"
    A, b = generate_lls_problem(n, d, noise_level, ill_conditioned=ill_conditioned)
    x0 = np.ones(d)

    # Objective function
    def g(x):
        return np.linalg.norm(A @ x - b)**2

    def h(t):
        return t**p

    def f(x):
        return h(g(x))

    # Gradient
    def gradg(x):
        return 2 * A.T @ (A @ x - b)

    def dh(t):
        return p * t**(p-1)

    def gradf(x):
        return dh(g(x)) * gradg(x)

    # LFSO
    def d2h(t):
        return p * (p-1) * t**(p-2)

    L_g = 2 * np.linalg.norm(A)**2
    mu_g = 2 * np.min(sp_linalg.svdvals(A))

    def lfso(x, R):
        c = L_g*R + np.linalg.norm(gradg(x))
        return d2h(0.5*c**2/mu_g) * c**2 + dh(0.5*c**2/mu_g) * L_g

    def Rk(x):
        return np.linalg.norm(gradg(x))

    return Objfun(f, gradf, lfso), x0, Rk


def linear_regression(n, d, noise_level, p, ill_conditioned=False):
    """
    Generate a test problem of the form

    f(x) = ||Ax-b||_{2p}^{2p} for some p>=1

    where Ax=b is an n*d linear system.
    """
    assert p >= 1, "Need p>=1 for composite problem"
    A, b = generate_lls_problem(n, d, noise_level, ill_conditioned=ill_conditioned)
    x0 = np.ones(d)

    # Objective function
    def f(x):
        return np.sum(np.abs(A @ x - b)**(2*p))

    # Gradient
    def gradf(x):
        return 2*p * A.T @ (A @ x - b)**(2*p-1)

    # LFSO
    Aop = np.linalg.norm(A, ord=2)
    max_row_norm = np.max(np.linalg.norm(A, axis=1))
    def lfso(x, R):
        if p > 1:
            return 2*p * (2*p-1) * Aop**2 * 2**(2*p-3) * (np.max((A@x-b)**(2*p-2)) + max_row_norm**(2*p-2) * R**(2*p-2))
        else:
            return 2 * Aop**2

    def Rk(x):
        return np.linalg.norm(A @ x - b, ord=np.inf)

    return Objfun(f, gradf, lfso), x0, Rk


def main():
    # objfun, x0, Rk = composite_linear_least_squares(10, 10, 0.1, 1, ill_conditioned=False)
    objfun, x0, Rk = linear_regression(10, 10, 0.1, 2, ill_conditioned=False)
    niters = 10000
    eta = 1.0

    xk = x0.copy()
    objfun.save_iterate(xk)
    print("  k        fk         ||gk||  ")
    for k in range(niters):
        fk = objfun.obj(xk)
        gk = objfun.grad(xk)
        if k % (niters // 10) == 0:
            print(f"{k:^8} {fk:^10.2e} {np.linalg.norm(gk):^10.2e}")
        Rk1 = Rk(xk)
        Rk2 = max(Rk1, eta * np.linalg.norm(gk) / objfun.lfso(xk, Rk1))
        Lk = objfun.lfso(xk, Rk2)
        xk = xk - (eta / Lk) * gk
        objfun.save_iterate(xk)

    fs, gs = objfun.get_history()

    plt.figure()
    plt.clf()
    plt.semilogy(fs, label='f(xk)')
    plt.semilogy(gs, label='g(xk)')
    plt.grid()
    plt.legend(loc='best')
    plt.show()
    print("Done")
    return


if __name__ == '__main__':
    main()
