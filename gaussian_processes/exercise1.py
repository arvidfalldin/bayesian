import argparse
from utils.gp_utils import (squared_exponential as se,
                            mater52, constant, identity)
import matplotlib.pyplot as plt
import numpy as np


def mode_0(args, kernel=se, kernel_kwargs={}, nugget=1e-6):
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')

    x = np.linspace(0, 1, args.n)

    for sample in range(args.num_samples):
        # Sample some kernel hyperparams
        sigma = 0.8*np.random.rand() + 0.1
        ell = 0.8*np.random.rand() + 0.2

        # Compute covariance kernel
        covar_0 = np.zeros((args.n, args.n))
        for i in range(args.n):
            covar_0[:, i] = se(x[i], x, sigma=sigma, ell=ell)
            covar_0[i, i] += nugget

        A = np.linalg.cholesky(covar_0)

        # Priors
        mu1 = constant(x)
        mu2 = identity(x)

        # Sample from standard multivariate normal
        z = np.random.randn(args.n)

        f1 = A @ z + mu1
        f2 = A @ z + mu2

        ax1.plot(x, f1, label=f'sigma={sigma:1.2f}, ell={ell:1.2f}')
        ax2.plot(x, f2, label=f'sigma={sigma:1.2f}, ell={ell:1.2f}')

    ax1.legend()
    ax2.legend()

    fig1.savefig('E1_gp_constant_prior.png')
    fig2.savefig('E1_gp_linear_prior.png')


def mode_1(args):
    pass


def mode_2(args):
    pass


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-points', '-n', type=int, default=100,
                        help='Number of points to sample')
    parser.add_argument('--num-samples', '-s', type=int, default=1,
                        help='Number of function to sample per setting')
    parser.add_argument('--mode', '-m', type=int, default=0,
                        help='which case to plot')
    args, _ = parser.parse_known_args()

    # Do some quick and dirty switch-like statements
    if args.mode == 0:
        mode_0(args)


