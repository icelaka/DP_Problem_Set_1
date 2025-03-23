"""

simulate.py
-----------
This code simulates the model.

"""

#%% Imports from Python
from numpy import cumsum,linspace,squeeze,where,zeros
from numpy.random import choice,rand,seed
from numpy.linalg import matrix_power
from types import SimpleNamespace
from numpy.random import normal

#%% Simulate the model.
def grow_economy(myClass):
    '''
    
    This function simulates the stochastic growth model.
    
    Input:
        myClass : Model class with parameters, grids, utility function, and policy functions.
        
    '''

    print('\n--------------------------------------------------------------------------------------------------')
    print('Simulate the Model')
    print('--------------------------------------------------------------------------------------------------\n')
    
    # Namespace for simulation.
    setattr(myClass,'sim',SimpleNamespace())
    sim = myClass.sim

    # Model parameters, grids and functions.
    
    par = myClass.par # Parameters.
    sol = myClass.sol # Policy functions.

    sigma = par.sigma # CRRA.
    util = par.util # Utility function.
    seed_sim = par.seed_sim # Seed for simulation.

    klen = par.klen # Capital grid size.
    Alen = par.Alen # Productivity grid size.
    kgrid = par.kgrid # Capital today (state).
    Agrid = par.Agrid[0] # Productivity today (state).
    pmat = par.pmat # Productivity today (state).
    
    yout = sol.y # Production function.
    kpol = sol.k # Policy function for capital.
    cpol = sol.c # Policy function for consumption.
    ipol = sol.i # Policy function for investment.

    T = par.T # Time periods.
    Asim = zeros(par.T*2) # Container for simulated productivity.
    ysim = zeros(par.T*2) # Container for simulated output.
    ksim = zeros(par.T*2) # Container for simulated capital stock.
    Tsim = zeros(par.T * 2)  # Container for simulated trade openness
    csim = zeros(par.T*2) # Container for simulated consumption.
    isim = zeros(par.T*2) # Container for simulated investment.
    usim = zeros(par.T*2) # Container for simulated utility.
            
    # Begin simulation.
    
    seed(seed_sim)

    pmat0 = matrix_power(pmat,1000)
    pmat0 = pmat0[0,:] # % Stationary distribution.
    cmat = cumsum(par.pmat,axis=1) # CDF matrix.

    A0_ind = choice(linspace(0,Alen,Alen,endpoint=False,dtype=int),1,p=pmat0) # Index for initial productivity.
    k0_ind = choice(linspace(0,klen,klen,endpoint=False,dtype=int),1) # Index for initial capital stock.

    Tsim[0] = par.trade_base
    ysim[0] = (
        Asim[0]
        * (1 + par.trade_lambda * Tsim[0])
        * yout[k0_ind, A0_ind]
        / (Agrid[A0_ind])
    )
    csim[0] = cpol[k0_ind,A0_ind] # Consumption in period 1 given k0 and A0.
    ksim[0] = kpol[k0_ind,A0_ind] # Capital choice for period 2 given k0.
    isim[0] = ipol[k0_ind,A0_ind] # Investment in period 1 given k0 and A0.
    usim[0] = util(csim[0],sigma) # Utility in period 1 given k0 and A0.

    A1_ind = where(rand(1)<=squeeze(cmat[A0_ind,:])) # Draw productivity for next period.
    At_ind = A1_ind[0][0]

    # Simulate endogenous variables.

    for j in range(1,T*2): # Time loop.
        kt_ind = where(ksim[j-1]==kgrid) # Capital choice in the previous period is the state today. Find where the latter is on the grid.
        Tsim[j] = (
            (1 + par.trade_trend_rate) * Tsim[j - 1]
            + par.trade_rho * (par.trade_trend_steady - Tsim[j - 1])
            + normal(0, par.trade_sigma)
        )
        # Tsim[j] = max(Tsim[j], 0.)
        print(Tsim[j])
        Asim[j] = Agrid[At_ind] # Productivity in period t.
        ysim[j] = Asim[j] * (1 + par.trade_lambda * Tsim[j]) * yout[kt_ind, At_ind]
        csim[j] = cpol[kt_ind,At_ind] # Consumption in period t.
        ksim[j] = kpol[kt_ind,At_ind] # Capital stock for period t+1.
        isim[j] = ipol[kt_ind,At_ind] # Investment in period t.
        usim[j] = util(csim[j],sigma) # Utility in period t.
        A1_ind = where(rand(1)<=squeeze(cmat[At_ind,:])) # Draw next state.
        At_ind = A1_ind[0][0] # State next period.

    # Burn the first half.
    sim.Asim = Asim[T:2*T+1] # Simulated productivity.
    sim.Tsim = Tsim[T : 2 * T + 1]  # Simulated trade openness
    sim.ysim = ysim[T:2*T+1] # Simulated output.
    sim.ksim = ksim[T:2*T+1] # Simulated capital choice.
    sim.csim = csim[T:2*T+1] # Simulated consumption.
    sim.isim = isim[T:2*T+1] # Simulated investment.
    sim.usim = usim[T:2*T+1] # Simulated utility.