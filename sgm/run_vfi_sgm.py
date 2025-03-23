"""

run_vfi_dgm.py
--------------
This code solves the stochastic growth model using value function iteration.

"""

# %% Import from folder
from model import planner
from solve import plan_allocations
from simulate import grow_economy
from my_graph import track_growth

# %% Import from Python and set project directory
import os

os.chdir("C:/Users/patata/Documents/Files/Code/New folder/sgm")
main = os.getcwd()
figout = main + "/output/figures"
os.makedirs(
    figout, exist_ok=True
)  # Create figout directory and parents if they don't exist


# %% Stochastic Growth Model.
benevolent_dictator = planner()

# Set the parameters, state space, and utility function.
benevolent_dictator.setup(
    main=main,
    figout=figout,
    beta=0.96,
    sigma=1.25,
    T = 100,
    trade_base=0.29,  # Baseline trade openness.
    trade_rho=0.793,  # Persistence
    trade_sigma=0.3,  # Std. dev
    trade_lambda=0.347,  # Elasticity
    trade_trend_rate = 0.005,
    trade_trend_steady = 1.12,
)  # You can set the parameters here or use the defaults.

# Solve the model.
plan_allocations(benevolent_dictator)  # Obtain the policy functions for capital.

# Simulate the model.
grow_economy(benevolent_dictator)  # Simulate forward in time.

# Graphs.
track_growth(benevolent_dictator)  # Plot policy functions and simulations.
