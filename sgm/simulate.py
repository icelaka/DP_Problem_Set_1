"""

simulate.py
-----------
This code simulates the model with trade openness \( T_t \) as a state variable.

"""

# %% Imports from Python
from numpy import cumsum, linspace, squeeze, where, zeros, argmin
from numpy.random import choice, rand, seed
from numpy.linalg import matrix_power
from types import SimpleNamespace


# %% Simulate the model.
def grow_economy(myClass):
    """
    Simulate the model with discretized \( T_t \).
    """

    print("\n--------------------------------------------------------------")
    print("Simulating the Model (3D State Space: k, A, T)")
    print("--------------------------------------------------------------\n")

    # Namespace for simulation results
    setattr(myClass, "sim", SimpleNamespace())
    sim = myClass.sim

    # Model parameters and policy functions
    par = myClass.par
    sol = myClass.sol
    kgrid = par.kgrid
    Agrid = par.Agrid[0]
    Tgrid = par.Tgrid[0]  # Trade openness grid
    klen, Alen, Tlen = par.klen, par.Alen, par.Tlen

    # Policy functions (3D: k, A, T)
    kpol = sol.k  # Shape: (klen, Alen, Tlen)
    cpol = sol.c
    ypol = sol.y

    # Transition matrices for A and T
    pmat = par.pmat  # Productivity transitions
    T_trans = par.T_trans  # Trade openness transitions
    T_trans_cdf = cumsum(T_trans, axis=1)  # CDF for T transitions

    # Simulation parameters
    T_total = par.T * 2  # Total periods (including burn-in)
    seed(par.seed_sim)

    # Initialize containers
    Asim = zeros(T_total)  # Productivity
    Tsim = zeros(T_total)  # Trade openness
    ksim = zeros(T_total)  # Capital
    ysim = zeros(T_total)  # Output
    csim = zeros(T_total)  # Consumption
    isim = zeros(T_total)  # Investment
    usim = zeros(T_total)  # Utility

    # Draw initial states (A, T) from stationary distribution
    pmat0_A = matrix_power(pmat, 1000)[0, :]  # Stationary dist for A
    pmat0_T = matrix_power(T_trans, 1000)[0, :]  # Stationary dist for T

    A0_ind = choice(range(Alen), p=pmat0_A)
    T0_ind = choice(range(Tlen), p=pmat0_T)
    k0_ind = argmin(abs(kgrid - par.kss))  # Start near steady state

    # Period 0
    Asim[0] = Agrid[A0_ind]
    Tsim[0] = Tgrid[T0_ind]  # Initialize T_t using discretized grid
    ksim[0] = kpol[k0_ind, A0_ind, T0_ind]
    ysim[0] = ypol[k0_ind, A0_ind, T0_ind]
    csim[0] = cpol[k0_ind, A0_ind, T0_ind]
    isim[0] = ksim[0] - (1 - par.delta) * kgrid[k0_ind]
    usim[0] = par.util(csim[0], par.sigma)

    # Track indices for A and T
    At_ind = A0_ind
    Tt_ind = T0_ind

    # Simulation loop
    for j in range(1, T_total):
        # Find current k index (closest grid point)
        kt_ind = argmin(abs(kgrid - ksim[j - 1]))

        # Draw next A and T indices using transitions
        A_rand = rand()
        At_ind = where(A_rand <= cumsum(pmat[At_ind, :]))[0][0]

        T_rand = rand()
        Tt_ind = where(T_rand <= T_trans_cdf[Tt_ind, :])[0][0]

        # Update states
        Asim[j] = Agrid[At_ind]
        Tsim[j] = Tgrid[Tt_ind]  # Update T_t from grid
        ksim[j] = kpol[kt_ind, At_ind, Tt_ind]
        ysim[j] = ypol[kt_ind, At_ind, Tt_ind]
        csim[j] = cpol[kt_ind, At_ind, Tt_ind]
        isim[j] = ksim[j] - (1 - par.delta) * kgrid[kt_ind]
        usim[j] = par.util(csim[j], par.sigma)

    # Burn first half
    T_burn = par.T
    sim.Asim = Asim[T_burn:]
    sim.Tsim = Tsim[T_burn:]  # Simulated trade openness
    sim.ksim = ksim[T_burn:]
    sim.ysim = ysim[T_burn:]
    sim.csim = csim[T_burn:]
    sim.isim = isim[T_burn:]
    sim.usim = usim[T_burn:]
