import numpy as np

def lagrange_polynomial_eval(X, Y, x):
    """Interpolates exactly through data"""
    X = np.array(X)
    Y = np.array(Y)

    if Y.ndim > 1:
        assert(1 in X.shape)
    N = X.size
    if Y.ndim == 1:
        Y = Y.reshape((1, N))
    else:
        assert(Y.shape[1] == N)

    res = 0
    for j in range(N):
        p = 1
        for i in range(N):
            if i != j:
                p = p * (x - X[i]) / (X[j] - X[i])

        res = res + p * Y[:, j]

    return res


