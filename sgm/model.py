"""

model.py
--------
This code sets up the model.

"""

#%% Imports from Python
from numpy import count_nonzero,exp,expand_dims,linspace,log,tile
from scipy import stats
from types import SimpleNamespace

#%% Deterministic Growth Model.
class planner():
    '''
    
    Methods:
        __init__(self,**kwargs) -> Set the household's attributes.
        setup(self,**kwargs) -> Sets parameters.
        
    '''
    
    #%% Constructor.
    def __init__(self,**kwargs):
        '''        
        
        This initializes the model.
        
        Optional kwargs:
            All parameters changed by setting kwarg.
            
        '''

        print('--------------------------------------------------------------------------------------------------')
        print('Model')
        print('--------------------------------------------------------------------------------------------------\n')
        print('   The model is the deterministic growth model and is solved via Value Function Iteration.')
        
        print('\n--------------------------------------------------------------------------------------------------')
        print("Analyzing Trade Policy Scenarios for Haiphong")
        # print('--------------------------------------------------------------------------------------------------\n')
        # print('   The household is infintely-lived.')
        # print('   It derives utility from consumption.')
        # print('    -> He/she can saves capital, which is used in production, for next period.')
        
    #%% Set up model.
    def setup(self,**kwargs):
        '''
        
        This sets the parameters and creates the grids for the model.
        
            Input:
                self : Model class.
                kwargs : Values for parameters if not using the default.
                
        '''
        
        # Namespace for parameters, grids, and utility function.
        setattr(self,'par',SimpleNamespace())
        par = self.par

        print('\n--------------------------------------------------------------------------------')
        print('Parameters:')
        print('--------------------------------------------------------------------------------\n')
        
        # Preferences.
        par.beta = 0.96 # Discount factor.
        par.sigma = 2.00 # CRRA.

        # Technology.

        par.alpha = 0.33 # Capital's share of income.
        par.delta = 0.05 # Depreciation rate of physical capital.

        par.sigma_eps = 0.07 # Std. dev of productivity shocks.
        par.rho = 0.85 # Persistence of AR(1) process.
        par.mu = 0.0 # Intercept of AR(1) process.

        # Simulation parameters.
        par.seed_sim = 2025 # Seed for simulation.
        par.T = 100 # Number of time periods.
        
        # Trade openness parameters
        par.trade_base = 0.3 # Baseline trade openness.
        par.trade_rho = 0.85 # Persistence of trade openness.
        par.trade_trend = 0.11 # Trend in trade openness.
        par.trade_sigma = 0.05 # Std. dev of trade openness shocks.
        par.trade_lambda = 0.2 # Elasticity
        par.trade_trend_rate = 0.05
        par.trade_trend_steady = 0.0212
        # par.prod_base = 1.0 # Baseline trade openness
        
        # Set up capital grid.
        par.kss = (par.alpha /((1.0/par.beta)-1+par.delta))**(1.0/(1.0-par.alpha)) # Steady state capital.
            
        par.klen = 300 # Grid size for k.
        par.kmax = 1.25*par.kss # Upper bound for k.
        par.kmin = 0.75*par.kss # Minimum k.
        
        # Discretize productivity.
        par.Alen = 7 # Grid size for A.
        par.m = 3.0 # Scaling parameter for Tauchen.
        
        # Update parameter values to kwarg values if you don't want the default values.
        for key,val in kwargs.items():
            setattr(par,key,val)
        
        assert par.main is not None
        assert par.figout is not None
        assert par.beta > 0 and par.beta < 1.00
        assert par.sigma >= 1.00
        assert par.trade_base >= 0.0
        assert par.alpha > 0 and par.alpha < 1.00
        assert par.delta >= 0 and par.delta <= 1.00
        assert par.sigma_eps > 0
        assert abs(par.rho) < 1
        assert par.Alen > 3
        assert par.m > 0.0
        assert par.klen > 5
        assert par.kmax > par.kmin
        
        # Set up capital grid.
        par.kgrid = linspace(par.kmin,par.kmax,par.klen) # Equally spaced, linear grid for k (and k').

        # Discretize productivity.
        Agrid,pmat = tauchen(par.mu,par.rho,par.sigma_eps,par.Alen,par.m) # Tauchen's Method to discretize the AR(1) process for log productivity.
        par.Agrid = exp(Agrid) # The AR(1) is in logs so exponentiate it to get A.
        par.pmat = pmat # Transition matrix.
    
        # Utility function.
        par.util = util
        
        print('beta: ',par.beta)
        print('sigma: ',par.sigma)
        print('kmin: ',par.kmin)
        print('kmax: ',par.kmax)
        print('trade_base: ',par.trade_base)
        print('trade_rho: ',par.trade_rho)
        print('trade_sigma: ',par.trade_sigma)
        print('trade_lambda: ',par.trade_lambda)
        print('alpha: ',par.alpha)
        print('delta: ',par.delta)
        print('sigma_eps: ',par.sigma_eps)
        print('rho: ',par.rho)
        print('mu: ',par.mu)

#%% CRRA Utility Function.
def util(c,sigma):

    # CRRA utility
    if sigma == 1:
        u = log(c) # Log utility.
    else:
        u = (c**(1-sigma))/(1-sigma) # CRRA utility.
    
    return u

#%% Tauchen's Method.
def tauchen(mu,rho,sigma,N,m):
    """
    
    This function discretizes an AR(1) process.
    
            y(t) = mu + rho*y(t-1) + eps(t), eps(t) ~ NID(0,sigma^2)
    
    Input:
        mu    : Intercept of AR(1).
        rho   : Persistence of AR(1).
        sigma : Standard deviation of error term.
        N     : Number of states.
        m     : Parameter such that m time the unconditional std. dev. of the AR(1) is equal to the largest grid point.
        
    Output:
        y    : Grid for the AR(1) process.
        pmat : Transition probability matrix.
        
    """
    
    #%% Construct equally spaced grid.
    
    ar_mean = mu/(1.0-rho) # The mean of a stationary AR(1) process is mu/(1-rho).
    ar_sd = sigma/((1.0-rho**2.0)**(1/2)) # The std. dev of a stationary AR(1) process is sigma/sqrt(1-rho^2)
    
    y1 = ar_mean-(m*ar_sd) # Smallest grid point is the mean of the AR(1) process minus m*std.dev of AR(1) process.
    yn = ar_mean+(m*ar_sd) # Largest grid point is the mean of the AR(1) process plus m*std.dev of AR(1) process.
     
    y,d = linspace(y1,yn,N,endpoint=True,retstep=True) # Equally spaced grid. Include endpoint (endpoint=True) and record stepsize, d (retstep=True).
    
    #%% Compute transition probability matrix from state j (row) to k (column).
    
    ymatk = tile(expand_dims(y,axis=0),(N,1)) # Container for state next period.
    ymatj = mu+rho*ymatk.T # States this period.
    
    # In the following, loc and scale are the mean and std used to standardize the variable. # For example, norm.cdf(x,loc=y,scale=s) is the standard normal CDF evaluated at (x-y)/s.
    pmat = stats.norm.cdf(ymatk,loc=ymatj-(d/2.0),scale=sigma)-stats.norm.cdf(ymatk,loc=ymatj+(d/2.0),scale=sigma) # Transition probabilities to state 2, ..., N-1.
    pmat[:,0] = stats.norm.cdf(y[0],loc=mu+rho*y-(d/2.0),scale=sigma) # Transition probabilities to state 1.
    pmat[:,N-1] = 1.0-stats.norm.cdf(y[N-1],loc=mu+rho*y+(d/2.0),scale=sigma) # Transition probabilities to state N.
    
    #%% Output.
    
    y = expand_dims(y,axis=0) # Convert 0-dimensional array to a row vector.
    
    if count_nonzero(pmat.sum(axis=1)<0.999999) > 0:
        raise Exception("Some columns of transition matrix don't sum to 1.") 

    return y,pmat