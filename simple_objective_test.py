#!/usr/bin/env python3

"""
Demonstrate LFSO on simple problems

f(x) = ||x||_{2}^{2p}
f(x) = ||x||_{2p}^{2p}

for p=1,2,3,...
"""
import numpy as np
import matplotlib.pyplot as plt

from main import Objfun, run_gd, run_gd_lfso

IMG_FOLDER = 'img'
FONT_SIZE = 'large'
IMG_FMT = 'pdf'


def generate_problem(d, p, norm='2'):
    assert p>=1, "Need p>=1, got p=%g" % p

    if norm == '2':
        # f(x) = (x_1^2 + ... + x_d^2)^p, written as f(x) = h(g(x)) for g(x)=||x||_2^2 and h(t)=t^p
        # Objective function
        def g(x):
            return np.linalg.norm(x) ** 2

        def h(t):
            return t ** p

        def f(x):
            return h(g(x))

        # Gradient
        def gradg(x):
            return 2 * x

        def dh(t):
            return p * t ** (p - 1)

        def gradf(x):
            return dh(g(x)) * gradg(x)

        # LFSO
        def d2h(t):
            if p >= 2:
                return p * (p - 1) * t ** (p - 2)
            else:
                return 0.0

        L_g = 2
        mu_g = 2

        def lfso(x, R):
            c = L_g * R + np.linalg.norm(gradg(x))
            return d2h(0.5 * c ** 2 / mu_g) * c ** 2 + dh(0.5 * c ** 2 / mu_g) * L_g

        def Rk(x):
            return np.linalg.norm(gradg(x))

    elif norm == '2p':
        def f(x):
            return np.sum(x**(2*p))  # x_1^{2p} + ... + x_d^{2p}

        def gradf(x):
            return 2*p * x**(2*p-1)

        # LFSO
        Aop = 1.0
        max_row_norm = 1.0

        def lfso(x, R):
            if p > 1:
                return 2*p * (2*p-1) * Aop**2 * 2**(2*p-3) * (np.max(x**(2*p-2)) + max_row_norm**(2*p-2) * R**(2*p-2))
            else:
                return 2*Aop**2

        def Rk(x):
            return np.linalg.norm(x, ord=np.inf)
    else:
        raise RuntimeError("Unknown norm = '%s'" % norm)

    objfun = Objfun(f, gradf, lfso)
    x0 = np.ones(d)
    return objfun, x0, Rk


def test_basic():
    d = 10
    p = 5
    # norm = '2'  # f(x) = ||x||_2^{2p}
    norm = '2p'  # f(x) = ||x||_{2p}^{2p}
    eta = 1.0
    niters = 1000

    objfun, x0, Rk = generate_problem(d=d, p=p, norm=norm)

    fs, gs = run_gd_lfso(objfun, x0, Rk, niters, eta, verbose=False)
    # print(fs)
    # print("*************")
    # print(gs)
    # exit()

    plt.figure()
    plt.clf()
    plt.figure()
    plt.clf()
    ax = plt.gca()
    plt.semilogy(fs, '-', linewidth=2, label=r'$f(x_k)$')
    plt.semilogy(gs, 'k--', linewidth=2, label=r'$\|\nabla f(x_k)\|$')
    ax.set_xlabel(r'Iteration $k$', fontsize=FONT_SIZE)
    ax.set_ylabel(r'$f(x_k)$, $\|\nabla f(x_k)\|$', fontsize=FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    plt.grid()
    plt.legend(loc='best')
    plt.show()
    return


def p_vary(d, norm, eta=1.0, niters=1000):
    ps = [1, 2, 3, 4, 5]

    plt.figure()
    plt.clf()
    plt.figure()
    plt.clf()
    ax = plt.gca()

    for p in ps:
        print(" - Running p = %g" % p)
        objfun, x0, Rk = generate_problem(d=d, p=p, norm=norm)
        fs, gs = run_gd_lfso(objfun, x0, Rk, niters, eta, verbose=False)
        plt.semilogy(gs / gs[0], '-', linewidth=2, label=r'$p = %g$' % p)
    ax.set_xlabel(r'Iteration $k$', fontsize=FONT_SIZE)
    ax.set_ylabel(r'$\|\nabla f(x_k)\| / \|\nabla f(x_0)\|$', fontsize=FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    plt.ylim(1e-15, 10)
    plt.grid()
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('%s/simple_norm%s_d%g.%s' % (IMG_FOLDER, norm, d, IMG_FMT), bbox_inches='tight')
    return


def main():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    d = 10
    # norm = '2'  # f(x) = ||x||_2^{2p}
    # norm = '2p'  # f(x) = ||x||_{2p}^{2p}
    for norm in ['2', '2p']:
        print("Using norm = %s" % norm)
        p_vary(d, norm, eta=1.0, niters=10000)
    print("Done")
    return


if __name__ == '__main__':
    main()
