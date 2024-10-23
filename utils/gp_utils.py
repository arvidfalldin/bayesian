import numpy as np
import math
from scipy.stats import norm as normal 

def squared_exponential(x1, x2, sigma=1.0, ell=1.0):
    return sigma**2*np.exp(-(x1-x2)**2 / (2*ell**2))


def mater52(x1, x2, sigma=1.0, ell=1.0):
    delta = x2 - x1
    r2 = delta**2
    return sigma * (1 + np.sqrt(5*r2)/ell + 5*r2/(3*ell**2))*np.exp(-np.sqrt(5*r2)/ell)


def mater12(x1, x2, theta=None, sigma=1.0):
    raise NotImplementedError
    return None


def mater32(x1, x2, theta=None, sigma=1.0):
    raise NotImplementedError
    return None


def identity(x):
    return x


def constant(x, value=0):
    return value*np.ones_like(x)


def relu(x):
    """
    The rectified linear unit
    y = max(0, x)
    """
    return x * step(x)


def expected_improvement2(mu, sigma, f_opt, ksi=0.0, gradient=False):
    """
    Expected improvement aquisition function  
    """
    delta = mu - f_opt

    # compute EI term by term
    ei = np.clip(delta, a_min=0, a_max=np.inf)
    ei += sigma*normal.pdf((delta - ksi) / (sigma + 1e-10))
    ei += -delta*normal.cdf((delta - ksi) / (sigma + 1e-10))
    return ei


def expected_improvement(x: np.ndarray,
                         x_obs: np.ndarray,
                         f_obs: np.ndarray,
                         kernel: callable,
                         mu0: callable,
                         ksi=0.0,
                         kernel_kwargs={},
                         gradient=False):
    """
    Expected improvement aquisition function

    Since the current implementation involves computing a matrix inverse,
    we try to be resourceful and use it to compute the EI gradient as well
    if needed.
    """
    pass

#     # Get the dimenensions
#     n_obs = len(x_obs)
#     n = len(x)

#     # Create the to covariance matrices
#     G = np.zeros(n_obs, n)
#     A = np.zeros(n_obs, n_obs)
#     b = f_obs - mu0(x_obs)

#     # Fill the covariance matrices
#     for i in range(n_obs):
#         G[i, :] = kernel(x_obs[i], x, **kernel_kwargs)
#         A[:, i] = kernel(x_obs[i], x_obs, **kernel_kwargs)


#     # Don't let Paolo see this...
#     A_inv = np.linalg.inv(A)

#     # Compute posterior mean
#     mu_n = G.T @ A_inv @ b + mu0(x)

#     # Compute posterior variance
#     sigma_n2 = kernel(x[0], x[0], **kernel_kwargs)
#     sigma_n2 += -G.T @ A_inv @ G.T

#     # Combine all observed data with new data
#     x_combined = np.concatenate([x, x_obs])
#     mu_combined = np.concatenate([mu_n, f_obs])
#     sigma_n2_combined = np.concatenate([sigma_n2, np.zeros_like(x_obs)])

#     # sort the x_values and apply permutation to all
#     perm = np.argsort(x_combined)
#     x_combined = x_combined[perm]
#     mu_combined = mu_combined[perm]
#     sigma_n2_combined = sigma_n2_combined[perm]

#     return x_combined, mu_combined, sigma_n2_combined


def compute_posterior(x: np.ndarray,
                      x_obs: np.ndarray,
                      f_obs: np.ndarray,
                      kernel: callable,
                      mu0: callable,
                      ksi=0.0,
                      nugget=1e-6,
                      kernel_kwargs={},
                      gradient=False):
    """
    Expected improvement aquisition function

    Since the current implementation involves computing a matrix inverse,
    we try to be resourceful and use it to compute the EI gradient as well
    if needed.
    """

    # Get the dimenensions
    n_obs = len(x_obs)
    n = len(x)

    # Create the to covariance matrices
    G = np.zeros((n_obs, n))
    A = np.zeros((n_obs, n_obs))
    b = f_obs - mu0(x_obs)

    # Fill the covariance matrices
    for i in range(n_obs):
        G[i, :] = kernel(x_obs[i], x, **kernel_kwargs)
        A[:, i] = kernel(x_obs[i], x_obs, **kernel_kwargs)
        A[i, i] += nugget

    # Don't let Paolo see this...
    A_inv = np.linalg.inv(A)

    # Compute posterior mean
    mu_n = G.T @ A_inv @ b + mu0(x)

    # Compute posterior variance
    sigma_n2 = np.ones(n)*kernel(x[0], x[0], **kernel_kwargs)
    sigma_n2 += nugget

    # Compute A @ G.T
    AGT = A_inv @ G
    # Step 2: Multiply element-wise G with the transpose of AGT and
    # sum over axis 1
    sigma_n2 += -np.einsum('ij,ji->i', G.T, AGT)

    # Combine all observed data with new data
    x_combined = np.concatenate([x, x_obs])
    mu_combined = np.concatenate([mu_n, f_obs])
    sigma_n2_combined = np.concatenate([sigma_n2, 0*x_obs])

    # sort the x_values and apply permutation to all
    perm = np.argsort(x_combined)
    x_combined = x_combined[perm]
    mu_combined = mu_combined[perm]
    sigma_n2_combined = sigma_n2_combined[perm]

    return x_combined, mu_combined, sigma_n2_combined


def step(x, x0=0):
    """
    Heaviside step function
    """
    return x >= x0


def ei_derivative(x, mu, sigma, f_opt, covar_func):
    
    delta = mu - f_opt
