import argparse
from utils.gp_utils import (squared_exponential as se,
                            mater52, constant, identity)
import matplotlib.pyplot as plt
import numpy as np


def mode_0(args, kernel=se, kernel_kwargs={}, nugget=1e-6):
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')

    x = np.linspace(0, 1, args.num_points)
    # Priors
    mu1 = constant(x)
    mu2 = identity(x)

    for sample in range(args.num_samples):
        # Sample some kernel hyperparams
        sigma = 0.1*np.random.rand() + 0.2
        ell = 0.1*np.random.rand() + 0.1

        # Compute covariance kernel
        covar_0 = np.zeros((args.num_points, args.num_points))
        for i in range(args.num_points):
            covar_0[:, i] = se(x[i], x, sigma=sigma, ell=ell)
            covar_0[i, i] += nugget

        A = np.linalg.cholesky(covar_0)

        # Sample from standard multivariate normal
        z = np.random.randn(args.num_points)

        f1 = A @ z + mu1
        f2 = A @ z + mu2

        ax1.plot(x, f1, label=f'sigma={sigma:1.2f}, ell={ell:1.2f}')
        ax2.plot(x, f2, label=f'sigma={sigma:1.2f}, ell={ell:1.2f}')

    ax1.plot(x, mu1, label='prior')
    ax2.plot(x, mu2, label='prior')

    ax1.legend()
    ax2.legend()

    fig1.savefig('E1_gp_constant_prior.png')
    fig2.savefig('E1_gp_linear_prior.png')


def mode_1(args, kernel=se, kernel_kwargs={}, nugget=1e-6):
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')

    x = np.linspace(-4, 4, args.num_points)

    # Prior
    mu = identity(x)

    sigmas = [1.0, 1.0, 0.2, 1.0]
    ells = [1.0, 0.5, 1.0, 5.0]

    for sample in range(args.num_samples):
        # Sample some kernel hyperparams
        sigma = sigmas[sample]
        ell = ells[sample]

        # Compute covariance kernel
        covar_a_0 = np.zeros((args.num_points, args.num_points))
        covar_b_0 = np.zeros((args.num_points, args.num_points))
        for i in range(args.num_points):
            covar_a_0[:, i] = se(x[i], x, sigma=sigma, ell=ell)
            covar_a_0[i, i] += nugget

            covar_b_0[:, i] = mater52(x[i], x, sigma=sigma, ell=ell)
            covar_b_0[i, i] += nugget

        A_a = np.linalg.cholesky(covar_a_0)
        A_b = np.linalg.cholesky(covar_b_0)

        # Sample from standard multivariate normal
        z = np.random.randn(args.num_points)

        f_a = A_a @ z + mu
        f_b = A_b @ z + mu

        l1, = ax1.plot(x, f_a, label=f'sigma={sigma:1.2f}, ell={ell:1.2f}, SE')
        ax2.plot(x, f_b, label=f'sigma={sigma:1.2f}, ell={ell:1.2f}, M52')

    ax1.plot(x, mu, label='prior')
    ax1.legend()

    ax2.plot(x, mu, label='prior')
    ax2.legend()

    fig1.savefig('E1_gp_kernel_se.png')
    fig2.savefig('E1_gp_kernel_mater52.png')

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    x = np.linspace(0, 3, 100)
    ax.plot(x, se(0, x), label='SE')
    ax.plot(x, mater52(0, x), label='Mater52')

    fig.savefig('kernels.png')


def mode_2(args, kernel=se, kernel_kwargs={}, nugget=1e-6):
    fig, ax = plt.subplots(2, 2, figsize=(16, 8), sharex=True, sharey=True)
    ax[1,0].set_xlabel('x')
    ax[1,1].set_xlabel('x')
    ax[0,0].set_ylabel('f(x)')
    ax[1,0].set_ylabel('f(x)')

    x = np.linspace(0, 1, args.num_points)

    # Priors
    mu = identity(x)

    nuggets = [-3, -4, -5, -6]

    for sample in range(args.num_samples):
        # Sample some kernel hyperparams
        sigma = 0.1*np.random.rand() + 0.2
        ell = 0.1*np.random.rand() + 0.1

        # Compute covariance kernel
        covar_0 = np.zeros((args.num_points, args.num_points))
        for i in range(args.num_points):
            covar_0[:, i] = se(x[i], x, sigma=sigma, ell=ell)

        # Sample from standard multivariate normal
        z = np.random.randn(args.num_points)

        for row in range(2):
            for col in range(2):
                nugget = 10**nuggets[2*row + col]
                covar = covar_0 + np.eye(args.num_points)*nugget
                A = np.linalg.cholesky(covar)
                f = A @ z + mu
                ax[row, col].plot(x, f, label=f'sigma={sigma:1.2f}, ell={ell:1.2f}, eps={nuggets[2*row + col]}')

    for row in range(2):
        for col in range(2):
            ax[row, col].plot(x, mu, label='prior')
            ax[row, col].legend()

    fig.savefig('E1_gp_nuggets.png')


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
    elif args.mode == 1:
        mode_1(args)
    elif args.mode == 2:
        mode_2(args)
