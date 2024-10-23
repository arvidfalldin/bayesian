import argparse
from utils.gp_utils import (squared_exponential as se,
                            mater52, constant, identity,
                            expected_improvement)

import matplotlib.pyplot as plt
import numpy as np
import random
import colorcet as cc

random.seed(4)


def mode_0(args):
    # Create the figure
    fig, ax = plt.subplots(2, 1, figsize=(16, 8))
    ax[0].set_xlabel('x')

    # Plot the unknown function that we are trying to find a extremum of
    x = np.linspace(-4, 4, args.num_points)

    def f_true(x):
        return 0.5*x + (np.sqrt(3*np.abs(x)/2) + 0.1)*np.sin(3*x/2) + np.cos(9*x/2)/(9/4*x**2 + 1)
    ax[0].plot(x, f_true(x), label='f_true(x)', color='k', linewidth=2)
    ax[0].set_ylim(-3, 3.5)

    x = np.linspace(-4, 4, args.num_points)

    index = random.sample(range(0, args.num_points), args.num_sample_points)
    x2 = x[index]
    f2 = f_true(x2)

    x1 = [xi for i, xi in enumerate(x) if i not in index]
    x1 = np.array(x1)

    x = np.concatenate([x1, x2])
    Sigma = np.zeros((args.num_points, args.num_points))

    sigma = 1.25
    ell = 0.4
    for i in range(args.num_points):
        Sigma[:, i] = mater52(x[i], x, sigma=sigma, ell=ell)
        Sigma[i, i] += 1e-6

    n1 = len(x1)
    n2 = args.num_points - n1

    Sigma11 = Sigma[:n1, :n1]
    Sigma12 = Sigma[:n1, n1:]
    Sigma21 = Sigma[n1:, :n1]
    Sigma22 = Sigma[n1:, n1:]

    Sigma_22_inv = np.linalg.inv(Sigma22)

    Sigma_1 = Sigma11 - Sigma12 @ Sigma_22_inv @ Sigma21

    # Sigma_1 = Sigma11 - Sigma12 @ Sigma_22_inv @ Sigma21
    mu1 = Sigma12 @ Sigma_22_inv @ f2


    mu = np.concatenate([mu1, f2])
    sigma = np.concatenate([np.diag(Sigma_1), 0*x2])

    mu = mu[x.argsort()]
    sigma = sigma[x.argsort()]
    x_sorted = x[x.argsort()]

    ax[0].plot(x_sorted, mu, color='blue', label='posterior mean', linewidth=2)
    ax[0].fill_between(x_sorted, mu - 1.96*sigma, mu + 1.96*sigma,
                       color='b', alpha=0.25, label='CI')
    ax[0].legend()

    # Expected improvement
    ei = expected_improvement(mu, sigma, f_opt=np.max(f2))
    ax[1].plot(x_sorted, ei, linewidth=2)   
    ax[1].set_ylim(-0.1, 1.0)

    fig.savefig('E3_EI.png')


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-points', '-n', type=int, default=200,
                        help='Number of points to use for plotting')
    parser.add_argument('--num-sample-points', '-nsp', type=int, default=5,
                        help='Number of points sampled')
    parser.add_argument('--num-sample-curves', '-nsc', type=int, default=5,
                        help='Number of points sampled')
    parser.add_argument('--mode', '-m', type=int, default=0,
                        help='which case to plot')
    args, _ = parser.parse_known_args()

    if args.mode == 0:
        mode_0(args)
