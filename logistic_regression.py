"""
Implement LFSO for logistic regression, possibly using SGD variant (with minibatching)
"""
import numpy as np
from scipy.special import logit, expit  # logit(p) = log(p/(1-p)), expit(t) = 1/(1+e^{-t})
from main import Objfun


def load_mnist():
    # Sample MNIST data of just 0 and 1 images
    data = np.loadtxt('lab13_mnist_train.csv', skiprows=1, delimiter=',')
    # First column is labels, remainder of columns are actual pixel information
    imgs = data[:, 1:]
    labels = data[:, 0]
    return imgs, np.array(labels, dtype=int)


def get_objfun():
    X, y = load_mnist()  # X is N*d
    ytilde = 2 * y - 1  # y in {0,1}, ytilde in {-1, 1}

    def f(w, batch_idx=None):
        if batch_idx is not None:
            return np.sum(np.log1p(np.exp(-ytilde[batch_idx] * (X[batch_idx,:] @ w))))
        else:
            # print(X.shape, w.shape, ytilde.shape)
            return np.sum(np.log1p(np.exp(-ytilde * (X @ w))))

    def gradf(w, batch_idx=None):
        mu = expit(X @ w)
        if batch_idx is not None:
            return X[batch_idx, :].T @ (mu[batch_idx] - y[batch_idx])
        else:
            return X.T @ (mu - y)

    def lfso(w, R, batch_idx=None):
        if batch_idx is not None:
            aR = np.exp(R * np.linalg.norm(X[batch_idx,:], axis=1))  # axis=1 is row-wise norm (i.e. for each xi)
            dw = np.exp(-X[batch_idx, :] @ w)
            Xnorm = np.linalg.norm(X[batch_idx, :], ord='fro')  # should be 2 norm but use Frobenius for speed
        else:
            aR = np.exp(R * np.linalg.norm(X, axis=1))  # axis=1 is row-wise norm (i.e. for each xi)
            dw = np.exp(-X @ w)
            Xnorm = np.linalg.norm(X, ord='fro')  # should be 2 norm but use Frobenius for speed
        Snorm = max(np.maximum(aR * dw / (1 + aR*dw)**2, aR * dw / (aR + dw)**2))
        if np.any(np.isnan(Snorm)):
            Snorm = 0.25
        if np.any((dw*aR >= 1) & (dw <= aR)):
            return max(Snorm, 0.25) * Xnorm**2
        else:
            return Snorm * Xnorm**2

    def Rk(wk, batch_idx=None):
        return np.linalg.norm(gradf(wk, batch_idx=batch_idx))

    w0 = np.zeros(X.shape[1])
    return f, gradf, lfso, Rk, w0


def run_gd(f, gradf, lfso, w0, Rk, niters, eta, use_lfso=True, verbose=False):
    iter_info = []

    wk = w0.copy()
    iter_info.append([0, f(wk), np.linalg.norm(gradf(wk))])
    if verbose:
        print("  k        fk         ||gk||  ")

    for k in range(niters):
        gk = gradf(wk)
        if verbose and k % (niters // 10) == 0:
            fk = f(wk)
            print(f"{k:^8} {fk:^10.2e} {np.linalg.norm(gk):^10.2e}")
        if use_lfso:
            Rk1 = Rk(wk)
            Rk2 = max(Rk1, eta * np.linalg.norm(gk) / lfso(wk, Rk1))
            Lk = lfso(wk, Rk2)
            print("Lk = %g" % Lk)
            wk = wk - (eta / Lk) * gk
        else:
            wk = wk - eta * gk
        iter_info.append([k+1, f(wk), np.linalg.norm(gradf(wk))])

    return np.array(iter_info)


def main():
    f, gradf, lfso, Rk, w0 = get_objfun()
    eta = 1.0
    niters = 100
    iter_info = run_gd(f, gradf, lfso, w0, Rk, niters, eta, use_lfso=True, verbose=True)
    print(iter_info)
    print("Done")
    return


if __name__ == '__main__':
    main()
