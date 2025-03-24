"""
my_graph.py
-----------
This code plots the model functions and simulations.
"""

# %% Imports from Python
import matplotlib.pyplot as plt
import numpy as np


# %% Plot the model functions and simulations.
def track_growth(myClass):
    """
    Plots policy functions and simulations after 3D to 2D aggregation.
    """

    # Model components
    par = myClass.par
    sol = myClass.sol
    sim = myClass.sim

    # =================================================================
    # Policy Functions (3D -> 2D by averaging)
    # =================================================================
    kgrid = par.kgrid

    # Average over A and T dimensions
    def avg_3D(arr):
        return np.mean(arr, axis=(1, 2)) if arr.ndim == 3 else arr

    # Production function
    plt.figure(1)
    plt.plot(kgrid, avg_3D(sol.y))
    plt.xlabel("$k_{t}$")
    plt.ylabel("$y_{t}$")
    plt.title("Production Function (Avg. over A,T)")
    plt.savefig(f"{par.figout}/ypol.png")
    plt.show()

    # Capital policy function
    plt.figure(2)
    plt.plot(kgrid, avg_3D(sol.k))
    plt.xlabel("$k_{t}$")
    plt.ylabel("$k_{t+1}$")
    plt.title("Capital Policy Function (Avg. over A,T)")
    plt.savefig(f"{par.figout}/kpol.png")
    plt.show()

    # Consumption policy function
    plt.figure(3)
    plt.plot(kgrid, avg_3D(sol.c))
    plt.xlabel("$k_{t}$")
    plt.ylabel("$c_{t}$")
    plt.title("Consumption Policy Function (Avg. over A,T)")
    plt.savefig(f"{par.figout}/cpol.png")
    plt.show()

    # Investment policy function
    plt.figure(4)
    plt.plot(kgrid, avg_3D(sol.i))
    plt.xlabel("$k_{t}$")
    plt.ylabel("$i_{t}$")
    plt.title("Investment Policy Function (Avg. over A,T)")
    plt.savefig(f"{par.figout}/ipol.png")
    plt.show()

    # Value function
    plt.figure(5)
    plt.plot(kgrid, avg_3D(sol.v))
    plt.xlabel("$k_{t}$")
    plt.ylabel("$V_t(k_t)$")
    plt.title("Value Function (Avg. over A,T)")
    plt.savefig(f"{par.figout}/vfun.png")
    plt.show()

    # =================================================================
    # Simulation Plots (1D time series)
    # =================================================================
    tgrid = np.arange(1, par.T + 1)

    # Output
    plt.figure(6)
    plt.plot(tgrid, sim.ysim)
    plt.xlabel("Time")
    plt.ylabel("$y^{sim}_t$")
    plt.title("Simulated Output")
    plt.savefig(f"{par.figout}/ysim.png")
    plt.show()

    # Capital
    plt.figure(7)
    plt.plot(tgrid, sim.ksim)
    plt.xlabel("Time")
    plt.ylabel("$k^{sim}_{t+1}$")
    plt.title("Simulated Capital")
    plt.savefig(f"{par.figout}/ksim.png")
    plt.show()

    # Consumption
    plt.figure(8)
    plt.plot(tgrid, sim.csim)
    plt.xlabel("Time")
    plt.ylabel("$c^{sim}_t$")
    plt.title("Simulated Consumption")
    plt.savefig(f"{par.figout}/csim.png")
    plt.show()

    # Investment
    plt.figure(9)
    plt.plot(tgrid, sim.isim)
    plt.xlabel("Time")
    plt.ylabel("$i^{sim}_t$")
    plt.title("Simulated Investment")
    plt.savefig(f"{par.figout}/isim.png")
    plt.show()

    # Utility
    plt.figure(10)
    plt.plot(tgrid, sim.usim)
    plt.xlabel("Time")
    plt.ylabel("$u^{sim}_t$")
    plt.title("Simulated Utility")
    plt.savefig(f"{par.figout}/usim.png")
    plt.show()

    # Productivity
    plt.figure(11)
    plt.plot(tgrid, sim.Asim)
    plt.xlabel("Time")
    plt.ylabel("$A^{sim}_t$")
    plt.title("Simulated Productivity")
    plt.savefig(f"{par.figout}/Asim.png")
    plt.show()

    # Trade Openness
    plt.figure(12)
    plt.plot(tgrid, sim.Tsim)
    plt.xlabel("Time")
    plt.ylabel("Trade Openness (T_t)")
    plt.title("Simulated Trade Openness")
    plt.savefig(f"{par.figout}/Tsim.png")
    plt.show()
