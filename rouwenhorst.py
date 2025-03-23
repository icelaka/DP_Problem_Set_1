#%% Import libraries
from numpy import count_nonzero, expand_dims, linspace
import numpy as np

# %% Rouwenhorst's Method for AR(1).
def rouwenhorst_ar1(mu, rho, sigma, N):
    ar_mean = mu / (1.0 - rho)  # The mean of a stationary AR(1) process is mu/(1-rho).
    ar_sd = sigma / (
        (1.0 - rho**2.0) ** (1 / 2)
    )  # The std. dev of a stationary AR(1) process is sigma/sqrt(1-rho^2)
    p = (1 - rho) / 2

    y1 = -(np.sqrt(N - 1) * (ar_sd))
    yn = np.sqrt(N - 1) * (ar_sd)

    y = linspace(
        y1, yn, N, endpoint=True
    )  # Equally spaced grid. Include endpoint (endpoint=True) and record stepsize, d (retstep=True).
    pmat = np.array([[p, 1 - p], [1 - p, p]])
    for i in range(N - 2):
        # Construct the transition matrix
        pmat = (
            p * np.pad(pmat, [(0, 1), (0, 1)], mode="constant")
            + (1 - p) * np.pad(pmat, [(0, 1), (1, 0)], mode="constant")
            + p * np.pad(pmat, [(1, 0), (1, 0)], mode="constant")
            + (1 - p) * np.pad(pmat, [(1, 0), (0, 1)], mode="constant")
        )

        pmat[1:-1] /= 2
        print(pmat)
    if count_nonzero(pmat.sum(axis=1) < 0.999) > 0:
        raise Exception("Some columns of transition matrix don't sum to 1.")

    y += ar_mean
    y = expand_dims(y, axis=0)

    print(
        f"return values of call with parameter rho = {rho}: \ny = \n{y}\npmat = \n{pmat}"
    )
    return y, pmat
