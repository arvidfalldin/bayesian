import numpy as np
import argparse
import matplotlib.pyplot as plt
import os


def exp2_kernel(x1, x2, sigma=1, ell=1):
    return sigma**2*.exp(-(x1-x2)**2 / (2*ell**2))


def identity(x):
    return x


def constant(x, value=0):
    return value*np.ones_like(x)


if __name__ == '__main__':

    kernel = exp2_kernel
    prior = constant

    n_samples = 4
    a, b = -4, 4
    n = 300
    nugget = 1e-7
    x = np.linspace(a, b, n)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')

    for sample in range(n_samples):
        sigma = 0.8*np.random.rand() + 0.1
        ell = 0.8*np.random.rand() + 0.2

        omega_0 = np.zeros((n, n))
        for i in range(n):
            omega_0[:, i] = kernel(x[i], x, sigma=sigma, ell=ell)
            omega_0[i, i] += nugget

        A = np.linalg.cholesky(omega_0)
        mu = prior(x)

        # Sample from standard multivariate normal
        z = np.random.randn(n)
        f = A @ z + mu

        ax.plot(x, f, label=f'sigma={sigma:1.2f}, ell={ell:1.2f}') 
    
    ax.legend()
    fig.savefig('gaussian_process_example.png')
