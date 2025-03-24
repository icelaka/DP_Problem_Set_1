"""

solve.py
--------
This code solves the model with vectorized operations for improved performance.

"""

# %% Imports from Python
import numpy as np
from numpy.linalg import norm
from types import SimpleNamespace
import time


# %% Solve the model using VFI.
def plan_allocations(myClass):
    """
    Solves the model using vectorized expectations and precomputed grids.
    """

    print("\n----------------------------------------------------------------")
    print("Solving the Model (Optimized 3D VFI)")
    print("----------------------------------------------------------------\n")

    # Namespace for optimal policy functions
    setattr(myClass, "sol", SimpleNamespace())
    sol = myClass.sol
    par = myClass.par

    t_start = time.time()

    # Grids
    kgrid = par.kgrid  # (klen,)
    Agrid = par.Agrid[0]  # (Alen,)
    Tgrid = par.Tgrid[0]  # (Tlen,)
    klen, Alen, Tlen = par.klen, par.Alen, par.Tlen

    # Broadcast to 3D grids
    kmat_3D = kgrid[:, None, None]  # (klen, 1, 1)
    Amat_3D = Agrid[None, :, None]  # (1, Alen, 1)
    Tmat_3D = Tgrid[None, None, :]  # (1, 1, Tlen)

    # Production function (precomputed)
    y0_3D = Amat_3D * (1 + par.trade_lambda * Tmat_3D) * (kmat_3D**par.alpha)
    c0_3D = y0_3D - par.delta * kmat_3D
    c0_3D = np.maximum(c0_3D, 0.0)  # Clip negative consumption

    # Initial value function
    v0 = par.util(c0_3D, par.sigma) / (1 - par.beta)
    v0[c0_3D <= 0.0] = -np.inf

    # Transition probability matrix (A x T)
    trans_probs = np.empty((Alen, Tlen, Alen * Tlen))
    for j in range(Alen):
        for m in range(Tlen):
            trans_probs[j, m] = np.outer(par.pmat[j], par.T_trans[m]).ravel()

    crit = 1e-6
    maxiter = 1000
    diff = np.inf
    iter = 0

    print(f"State space size: {klen}x{Alen}x{Tlen} = {klen * Alen * Tlen:,} states")
    print("Iterating...")

    while (diff > crit) and (iter < maxiter):
        v1 = np.full((klen, Alen, Tlen), -np.inf)
        k1 = np.zeros((klen, Alen, Tlen))

        # Reshape value function for vectorized operations
        v0_flat = v0.reshape(klen, -1)  # (klen, Alen*Tlen)

        # Main loop vectorization
        for p in range(klen):  # Current capital index
            k = kgrid[p]

            # Compute candidate capital (vectorized)
            kprime = kgrid  # (klen,)
            inv = kprime - (1 - par.delta) * k  # (klen,)
            c_candidates = y0_3D[p] - inv[:, None, None]  # (klen, Alen, Tlen)
            c_candidates = np.maximum(c_candidates, 0.0)

            # Compute utility (vectorized)
            util_grid = par.util(c_candidates, par.sigma)  # (klen, Alen, Tlen)

            # Compute expectations (vectorized matrix multiplication)
            exp_v = util_grid + par.beta * (
                v0_flat @ trans_probs.reshape(-1, Alen * Tlen)
            ).reshape(klen, Alen, Tlen)

            # Find optimal policy
            max_indices = np.argmax(exp_v, axis=0)
            v1[p] = np.take_along_axis(exp_v, max_indices[None], axis=0).squeeze(0)
            k1[p] = kgrid[max_indices]

        # Convergence check
        diff = np.max(np.abs(v1 - v0))
        v0 = v1.copy()
        iter += 1

        if iter % 10 == 0:
            print(
                f"Iter {iter:3d} | Max diff: {diff:.2e} | Time: {time.time() - t_start:.1f}s"
            )

    sol.k = k1
    sol.y = y0_3D
    sol.i = k1 - (1 - par.delta) * kmat_3D
    sol.c = np.maximum(y0_3D - sol.i, 0.0)
    sol.v = v1

    print(f"\nConverged in {iter} iterations")
    print(f"Total runtime: {time.time() - t_start:.1f} seconds")
