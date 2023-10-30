#!/usr/bin/env python
"""
Demonstration of LFSO utility on least-squares and similar problems
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sp_linalg

IMG_FOLDER = 'img'
FONT_SIZE = 'large'
IMG_FMT = 'pdf'


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


def generate_lls_problem(n, d, noise_level, cond_num=1000):
    # Generate a random linear least-squares problem of size n*d

    # Form the matrix to have specific condition number
    if n >= d:
        U = np.linalg.qr(np.random.randn(n, d), mode='reduced')[0]
        S = np.diag(np.linspace(1, cond_num, d))
        VT = np.linalg.qr(np.random.randn(d, d), mode='reduced')[0]
    else:
        U = np.linalg.qr(np.random.randn(n, n), mode='reduced')[0]
        S = np.diag(np.linspace(1, cond_num, n))
        VT = np.linalg.qr(np.random.randn(d, n), mode='reduced')[0].T  # flip to make n*d

    A = U @ S @ VT

    # RHS is such that ones(d) is a good estimate
    b = A @ np.ones((d,)) + noise_level*np.random.randn(n)
    return A, b


def composite_linear_least_squares(n, d, noise_level, p, cond_num=1000):
    """
    Generate a test problem of the form

    f(x) = ||Ax-b||_{2}^{2p} for some p>=1

    where Ax=b is an n*d linear system.

    This problem is written in the form f(x) = h(g(x)) where
    - g(x) = ||Ax-b||_{2}^{2}
    - h(t) = t^p
    """
    assert p >= 1, "Need p>=1 for composite problem"
    A, b = generate_lls_problem(n, d, noise_level, cond_num=cond_num)
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


def linear_regression(n, d, noise_level, p, cond_num=1000):
    """
    Generate a test problem of the form

    f(x) = ||Ax-b||_{2p}^{2p} for some p>=1

    where Ax=b is an n*d linear system.
    """
    assert p >= 1, "Need p>=1 for composite problem"
    A, b = generate_lls_problem(n, d, noise_level, cond_num=cond_num)
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


def get_problem(prob_str):
    np.random.seed(0)

    if prob_str.startswith('linreg_simple_p'):  # e.g. linreg_simple_p1 for LLS problem
        # Simple linear least-squares (overdetermined)
        n, d, noise_level, cond_num = 1000, 100, 0.1, 100
        p = int(prob_str.replace('linreg_simple_p', ''))
        objfun, x0, Rk = linear_regression(n, d, noise_level, p, cond_num=cond_num)
    elif prob_str.startswith('linreg_simple_consistent_p'):  # e.g. linreg_simple_consistent_p1 for LLS problem
        # Simple linear least-squares (determined)
        n, d, noise_level, cond_num = 100, 100, 0.1, 100
        p = int(prob_str.replace('linreg_simple_consistent_p', ''))
        objfun, x0, Rk = linear_regression(n, d, noise_level, p, cond_num=cond_num)
    elif prob_str.startswith('comp_simple_p'):  # e.g. comp_simple_p1 for LLS problem
        n, d, noise_level, cond_num = 1000, 100, 0.1, 100
        p = int(prob_str.replace('comp_simple_p', ''))
        objfun, x0, Rk = composite_linear_least_squares(n, d, noise_level, p, cond_num=cond_num)
    elif prob_str.startswith('comp_simple_consistent_p'):  # e.g. comp_simple_consistent_p1 for LLS problem
        n, d, noise_level, cond_num = 100, 100, 0.1, 100
        p = int(prob_str.replace('comp_simple_consistent_p', ''))
        objfun, x0, Rk = composite_linear_least_squares(n, d, noise_level, p, cond_num=cond_num)
    else:
        raise RuntimeError("Unknown prob_str: %s" % prob_str)
    return objfun, x0, Rk


def run_gd(objfun, x0, niters, eta, verbose=False):
    objfun.clear_history()

    xk = x0.copy()
    objfun.save_iterate(xk)
    if verbose:
        print("  k        fk         ||gk||  ")

    for k in range(niters):
        gk = objfun.grad(xk)
        if verbose and k % (niters // 10) == 0:
            fk = objfun.obj(xk)
            print(f"{k:^8} {fk:^10.2e} {np.linalg.norm(gk):^10.2e}")
        xk = xk - eta * gk
        objfun.save_iterate(xk)

    fs, gs = objfun.get_history()
    objfun.clear_history()
    return fs, gs


def run_gd_lfso(objfun, x0, Rk, niters, eta, verbose=False):
    objfun.clear_history()

    xk = x0.copy()
    objfun.save_iterate(xk)
    if verbose:
        print("  k        fk         ||gk||  ")

    for k in range(niters):
        gk = objfun.grad(xk)
        if verbose and k % (niters // 10) == 0:
            fk = objfun.obj(xk)
            print(f"{k:^8} {fk:^10.2e} {np.linalg.norm(gk):^10.2e}")
        Rk1 = Rk(xk)
        Rk2 = max(Rk1, eta * np.linalg.norm(gk) / objfun.lfso(xk, Rk1))
        Lk = objfun.lfso(xk, Rk2)
        xk = xk - (eta / Lk) * gk
        objfun.save_iterate(xk)

    fs, gs = objfun.get_history()
    objfun.clear_history()
    return fs, gs


def strongly_convex_compare(consistent):
    # Quadratic LLS problem, so just compare LFSO with correct 1/L stepsize
    if consistent:
        prob_str = 'linreg_simple_consistent_p1'
    else:
        prob_str = 'linreg_simple_p1'

    objfun, x0, Rk = get_problem(prob_str)
    niters = 20000
    eta_gd = 1 / objfun.lfso(x0, 1)

    print(prob_str)
    print(" - Running GD")
    fs_gd, gs_gd = run_gd(objfun, x0, niters, eta_gd, verbose=False)
    print(" - Running LFSO")
    fs_lfso, gs_lfso = run_gd_lfso(objfun, x0, Rk, niters, eta=1.0, verbose=False)
    print(" - Plotting")

    # Plot gradient decrease
    plt.figure()
    plt.clf()
    ax = plt.gca()
    plt.semilogy(gs_gd, 'b-', linewidth=2, label=r'GD')
    plt.semilogy(gs_lfso, 'r--', linewidth=2, label=r'LFSO')
    ax.set_xlabel(r'Iteration $k$', fontsize=FONT_SIZE)
    ax.set_ylabel(r'$\|\nabla f(x_k)\|$', fontsize=FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    plt.grid()
    plt.legend(loc='best')
    plt.savefig('%s/%s_niters%g_grad.%s' % (IMG_FOLDER, prob_str, niters, IMG_FMT), bbox_inches='tight')

    # Plot objective value
    plt.figure()
    plt.clf()
    ax = plt.gca()
    plt.semilogy(fs_gd, 'b-', linewidth=2, label=r'GD')
    plt.semilogy(fs_lfso, 'r--', linewidth=2, label=r'LFSO')
    ax.set_xlabel(r'Iteration $k$', fontsize=FONT_SIZE)
    ax.set_ylabel(r'$f(x_k)$', fontsize=FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    plt.grid()
    plt.legend(loc='best')
    plt.savefig('%s/%s_niters%g_obj.%s' % (IMG_FOLDER, prob_str, niters, IMG_FMT), bbox_inches='tight')
    return


def flat_convex_compare(consistent, p):
    # LLS problem with flat minimum, try different stepsizes
    assert p > 1, "Supposed to be for p>1 (p=1 is strongly_convex_compare)"
    if consistent:
        prob_str = 'linreg_simple_consistent_p%g' % p
    else:
        prob_str = 'linreg_simple_p%g' % p

    objfun, x0, Rk = get_problem(prob_str)
    niters = 50000
    etas_gd = [(1e-9, '10^{-9}'), (1e-7, '10^{-7}'), (1e-5, '10^{-5}')]  #, (1e-3, '10^{-3}')]

    print(prob_str)
    gd_results = {}
    # for eta_gd, eta_gd_str in etas_gd:
    #     print(" - Running GD (eta = %g)" % eta_gd)
    #     gd_results[eta_gd_str] = run_gd(objfun, x0, niters, eta_gd, verbose=False)
    print(" - Running LFSO")
    fs_lfso, gs_lfso = run_gd_lfso(objfun, x0, Rk, niters, eta=1.0, verbose=False)
    print(" - Plotting")

    # Plot gradient decrease
    plt.figure()
    plt.clf()
    ax = plt.gca()
    for eta_gd, eta_gd_str in etas_gd:
        gs_gd = gd_results[eta_gd_str][1]
        plt.semilogy(gs_gd, '-', linewidth=2, label=r'GD ($\eta=%s$)' % eta_gd_str)
    plt.semilogy(gs_lfso, 'k--', linewidth=2, label=r'LFSO')
    ax.set_xlabel(r'Iteration $k$', fontsize=FONT_SIZE)
    ax.set_ylabel(r'$\|\nabla f(x_k)\|$', fontsize=FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    plt.grid()
    plt.legend(loc='best')
    plt.savefig('%s/%s_niters%g_grad.%s' % (IMG_FOLDER, prob_str, niters, IMG_FMT), bbox_inches='tight')

    # Plot objective value
    plt.figure()
    plt.clf()
    ax = plt.gca()
    for eta_gd, eta_gd_str in etas_gd:
        fs_gd = gd_results[eta_gd_str][0]
        plt.semilogy(fs_gd, '-', linewidth=2, label=r'GD ($\eta=%s$)' % eta_gd_str)
    plt.semilogy(fs_lfso, 'k--', linewidth=2, label=r'LFSO')
    ax.set_xlabel(r'Iteration $k$', fontsize=FONT_SIZE)
    ax.set_ylabel(r'$f(x_k)$', fontsize=FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    plt.grid()
    plt.legend(loc='best')
    plt.savefig('%s/%s_niters%g_obj.%s' % (IMG_FOLDER, prob_str, niters, IMG_FMT), bbox_inches='tight')
    return


def lfso_p_compare(consistent=True, p=1, niters=50000):
    # Try LFSO for linreg with different p choices
    if consistent:
        prob_str = 'linreg_simple_consistent_p%g' % p
    else:
        prob_str = 'linreg_simple_p%g' % p
    objfun, x0, Rk = get_problem(prob_str)
    print(" - Running LFSO (p = %g)" % p)
    fs_lfso, gs_lfso = run_gd_lfso(objfun, x0, Rk, niters, eta=1.0, verbose=False)
    print(" - Plotting")

    # Plot gradient decrease
    plt.figure()
    plt.clf()
    ax = plt.gca()
    plt.semilogy(gs_lfso, '-', linewidth=2, label=r'$p=%g$' % p)
    ax.set_xlabel(r'Iteration $k$', fontsize=FONT_SIZE)
    ax.set_ylabel(r'$\|\nabla f(x_k)\|$', fontsize=FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    ax.grid()
    # plt.legend(loc='best')
    plt.savefig('%s/lfso_p%g_grad.%s' % (IMG_FOLDER, p, IMG_FMT), bbox_inches='tight')

    # Plot objective value
    plt.figure()
    plt.clf()
    ax = plt.gca()
    plt.semilogy(fs_lfso, '-', linewidth=2, label=r'$p=%g$' % p)
    ax.set_xlabel(r'Iteration $k$', fontsize=FONT_SIZE)
    ax.set_ylabel(r'$f(x_k)$', fontsize=FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    plt.grid()
    # plt.legend(loc='best')
    plt.savefig('%s/lfso_p%g_obj.%s' % (IMG_FOLDER, p, IMG_FMT), bbox_inches='tight')
    return


def main():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # strongly_convex_compare(consistent=False)
    # strongly_convex_compare(consistent=True)
    # flat_convex_compare(consistent=True, p=2)
    for p, niters in [(2, int(2e7))]:  #, (3, int(2e6))]:  # (1, 100000)
        lfso_p_compare(consistent=True, p=p, niters=niters)
    print("Done")
    return


if __name__ == '__main__':
    main()
