import numpy as np


def LagrangePolynomialEval(X, Y, x):
    """Interpolates exactly through data"""
    if isinstance(X, list):
        X = np.array(X)
    if isinstance(Y, list):
        Y = np.array(Y)
    if len(Y.shape) > 1:
        assert 1 in X.shape
    N = np.prod(X.shape)
    if len(Y.shape) == 1:
        Y = Y.reshape((1, N))
    else:
        assert(Y.shape[1] == N)

    res = 0

    for j in range(N):
        p = 1
        for i in range(N):
            if i!=j:
               p = p*(x-X[i])/(X[j]-X[i])

        res = res+p*Y[:, j]
    return res


def lagrange_polynomial_eval_numeric(X, Y, x):
    """Interpolates exactly through data points. This function only works
    with numerical arguments such as lists or Numpy arrays.

    Arguments
    ---------
        X : (N, ) array
            x-axis data points
        Y : (N, ) or (N, ny) array
            y-axis data points (possibly more than one dimension)
        x : (nx) array
            points on x axis at which to evaluate polynomial.

    """
    X = np.array(X)
    Y = np.array(Y)
    x = np.array(x)

    assert X.ndim == 1
    N = X.size
    if Y.ndim == 1:
        Y = Y.reshape((N, 1))
    assert x.ndim == 1
    nx = x.size

    ii, jj = np.indices((N, N))
    mask = ii != jj
    num = np.where(np.expand_dims(mask, 2), x - np.expand_dims(X[ii], 2), 1.)
    den = np.expand_dims(np.where(mask, X[jj] - X[ii], 1.), 2)
    p = np.prod(num / den, axis=0)
    y = np.matmul(p.T, Y)

    return y
