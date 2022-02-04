import enum
import math
import numpy as np
from scipy.linalg import expm
import argparse

from problems import get_problem


class Solution(object):
    def __init__(self, dim):
        self.f = float("nan")
        self.x = np.zeros([dim, 1])
        self.z = np.zeros([dim, 1])


class LRType(enum.Enum):
    ADAPTIVE = "adaptive"
    FIXED = "fixed"

    def __repr__(self) -> str:

        return str(self)


def main(**params):
    obj_func = get_problem(params["function"])
    dim = params["dim"]
    lamb = params["lamb"]
    mean = params["mean"]
    sigma = params["sigma"]
    seed = params["seed"]
    flg_lr = params["flg_lr"]
    multiply_lr_coef = params["multiply_lr_coef"]
    alpha = params["alpha"]
    beta = params["beta"]

    max_evals = int(5 * dim * 1e4)
    criterion = 1e-8

    np.random.seed(seed)
    # constant
    wrh = math.log(lamb / 2.0 + 1.0) - np.log(np.arange(1, lamb + 1))
    w_hat = np.maximum([0 * lamb], wrh)
    w = w_hat / sum(w_hat) - 1.0 / lamb

    eta_m = 1.0
    eta_B = 3.0 * (3.0 + np.log(dim)) / 5.0 / dim / np.sqrt(dim)

    if flg_lr == "fixed":
        eta_B *= multiply_lr_coef
        if "fixed_lr" in params:
            eta_B = params["fixed_lr"]

    eta_sigma = eta_B
    I = np.eye(dim, dtype=float)
    # dynamic
    mean = np.array([mean] * dim).reshape(dim, 1)
    B = np.eye(dim, dtype=float)
    evals = 0
    g = 0
    best = np.inf
    sols = [Solution(dim) for _ in range(lamb)]

    # learning rate adaptation
    mu_w = 1 / np.sum(w ** 2, axis=0)
    eta_sigma_min = eta_sigma
    eta_B_min = eta_B
    eta_sigma_max = eta_B_max = 1.0
    pm = np.zeros([dim, 1])
    pS = np.zeros([dim, dim])
    gamma = 0.0

    while evals < max_evals:
        g += 1

        nan_exists = False
        for i in range(lamb):
            sols[i].z = np.random.randn(dim, 1)
            sols[i].x = mean + sigma * B.dot(sols[i].z)
            sols[i].f = obj_func(sols[i].x)
            if sols[i].f is np.nan:
                nan_exists = True
        evals += lamb

        if nan_exists:
            break

        sols = sorted(sols, key=lambda s: s.f)
        best = min(best, sols[0].f)
        print("evals:{}, best:{}".format(evals, best)) if g % 1000 == 0 else None

        if sols[0].f < criterion:
            print(f"optimal x:{sols[0].x}")
            break

        # natural gradient
        G_delta = np.sum([w[i] * sols[i].z for i in range(lamb)], axis=0)
        G_M = np.sum(
            [w[i] * (np.outer(sols[i].z, sols[i].z) - I) for i in range(lamb)], axis=0
        )
        G_sigma = G_M.trace() / dim
        G_B = G_M - G_sigma * I

        # calculate delta for learning rate adaptation
        delta_m = -np.copy(mean)
        delta_cov = -(sigma ** 2) * B.dot(B.T)
        cov = (sigma ** 2) * B.dot(B.T)
        e, v = np.linalg.eigh(cov)  # remove the effect of sigma
        diag_sqrt_eig = np.diag(np.sqrt(e))
        inv_sqrt_cov = v @ np.linalg.inv(diag_sqrt_eig) @ v.T

        # update parameters
        mean += eta_m * sigma * np.dot(B, G_delta)
        sigma *= math.exp((eta_sigma / 2.0) * G_sigma)
        B = B.dot(expm((eta_B / 2.0) * G_B))

        # update delta
        delta_m += mean
        delta_cov += (sigma ** 2) * B.dot(B.T)
        approx_cov = (
            eta_B ** 2
            / 2.0
            * (1.0 + 4 * (eta_sigma ** 2) / (dim * mu_w))
            * (dim ** 2 + dim - 2.0)
            + eta_sigma ** 2
        ) / mu_w
        approx = approx_cov  # considering only covariance matrix, not mean

        # update evolution path in parameter space
        pm = (1 - beta) * pm + np.sqrt(beta * (2.0 - beta)) / np.sqrt(
            approx
        ) * inv_sqrt_cov.dot(delta_m)
        new_cov = (sigma ** 2) * B.dot(B.T)
        normalized_new_cov = inv_sqrt_cov.dot(new_cov.dot(inv_sqrt_cov))
        pS = (1 - beta) * pS + np.sqrt(beta * (2.0 - beta)) / np.sqrt(approx) * (
            normalized_new_cov - np.eye(dim)
        )
        square_ptheta_norm = np.trace(pS.dot(pS)) / 2.0

        gamma = (1 - beta) ** 2 * gamma + beta * (2 - beta)

        if LRType(flg_lr) == LRType.ADAPTIVE:
            eta_sigma = eta_sigma * np.exp(beta * (square_ptheta_norm / alpha - gamma))
            eta_sigma = min(max(eta_sigma, eta_sigma_min), eta_sigma_max)

            eta_B = eta_B * np.exp(beta * (square_ptheta_norm / alpha - gamma))
            eta_B = min(max(eta_B, eta_B_min), eta_B_max)


    print(evals, sols[0].f, seed)
    return evals, sols[0].f


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("function", type=str)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--lamb", type=int, default=30)
    parser.add_argument("--mean", type=float, default=3.0)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--flg_lr", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--multiply_lr_coef", type=float, default=1.0)
    # hyperparameters for learning rate adaptation
    parser.add_argument("--alpha", type=float, default=1.3)
    parser.add_argument("--beta", type=float, default=0.2)
    main(**vars(parser.parse_args()))
