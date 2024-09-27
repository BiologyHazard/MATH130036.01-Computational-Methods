from typing import Literal

import numpy as np
import scipy.special as sps


def make_polynomial_interpolation_equal_step(start, stop, n, y):
    """等距节点多项式插值"""
    h = (stop - start) / n
    delta_n_fk = np.zeros((n+1, n+1))
    """delta_n_fk[n, k] = Δ^n f_k"""
    delta_n_fk[0, :] = y
    for i in range(0, n):
        delta_n_fk[i+1, :n-i] = np.diff(delta_n_fk[i, :n-i+1])

    def p(x):
        """插值多项式"""
        t = (x - start) / h
        return sum(sps.binom(t, i) * delta_n_fk[i, 0] for i in range(n+1))

    return p


def make_cubic_spline_interpolation(x, y, type_: Literal["D1", "D2", "d1", "d2"] = "D2", s0=0, sn=0):
    """三次样条插值"""
    if len(x) != len(y):
        raise ValueError("x, y must have the same length")

    type_ = type_.upper()  # type: ignore
    if type_ not in ["D1", "D2"]:
        raise ValueError("type_ must be 'D1' or 'D2'")

    n = len(x) - 1
    h = np.diff(x)
    mu = np.zeros(n+1)
    mu[1:-1] = h[:-1] / (h[:-1] + h[1:])
    lambda_ = np.zeros(n+1)
    lambda_[1:-1] = h[1:] / (h[:-1] + h[1:])
    df = np.diff(y) / h
    """df[i] = f[x_i, x_{i+1}]"""
    d = np.zeros(n+1)
    d[1:-1] = 6 * np.diff(df) / (h[:-1] + h[1:])
    """d[i] = 6 * f[x_i, x_{i+1}, x_{i+2}]"""
    if type_ == "D1":
        lambda_[0] = 1
        d[0] = 6 * (df[0] - s0) / h[0]
        mu[-1] = 1
        d[-1] = 6 * (sn - df[-1]) / h[-1]
    elif type_ == "D2":
        lambda_[0] = 0
        d[0] = 2 * s0
        mu[-1] = 0
        d[-1] = 2 * sn

    A = np.zeros((n+1, n+1))
    for i in range(0, n+1):
        if i-1 >= 0:
            A[i, i-1] = mu[i]
        A[i, i] = 2
        if i+1 <= n:
            A[i, i+1] = lambda_[i]

    m = np.linalg.solve(A, d)
    """m[i] = f''(x_i)"""

    x_points = x

    def p(x):
        """插值函数"""
        j = np.searchsorted(x_points, x, side="right") - 1
        j = np.clip(j, 0, n-1)
        return (m[j] * (x_points[j+1] - x)**3 / (6 * h[j])
                + m[j+1] * (x - x_points[j])**3 / (6 * h[j])
                + (y[j] - m[j] * h[j]**2 / 6) * (x_points[j+1] - x) / h[j]
                + (y[j+1] - m[j+1] * h[j]**2 / 6) * (x - x_points[j]) / h[j])

    return p
