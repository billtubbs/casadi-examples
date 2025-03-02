import numpy as np
import matplotlib.pyplot as plt
import os


def make_uxplot(t, U, X, t_label='Time, $t$', x_titles=None, u_titles=None, filename=None, plot_dir=None):

    t = np.array(t)
    U = np.array(U)
    assert U.ndim == 2
    X = np.array(X)
    assert X.ndim == 2
    nx, nu = X.shape[1], U.shape[1]
    if x_titles is None:
        x_titles = [None] * nx
    if u_titles is None:
        u_titles = [None] * nu
    if nx > 1:
        x_labels = [f'$x_{i}$' for i in range(nx)]
    else:
        x_labels = [f'$x$']
    if nu > 1:
        u_labels = [f'$u_{i}$' for i in range(nu)]
    else:
        u_labels = [f'$u$']

    fig, axes = plt.subplots(nx + nu, 1, sharex=True)

    for i, (ax, title, label) in enumerate(zip(axes[:nx], x_titles, x_labels)):
        ax.plot(t, X[:, i], '.-')
        ax.grid()
        ax.set_ylabel(label)
        ax.set_title(title)

    for i, (ax, title, label) in enumerate(zip(axes[nx:], u_titles, u_labels)):
        ax.plot(t, U[:, i], '.-')
        ax.grid()
        ax.set_ylabel(label)
        ax.set_title(title)

    axes[-1].set_xlabel(t_label) 
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(plot_dir, filename))
    
    return fig, axes
