"""

my_graph.py
-----------
This code plots the value and policy functions.

"""

#%% Imports from Python
from matplotlib.pyplot import figure,plot,xlabel,ylabel,title,savefig
from numpy import linspace

#%% Plot the model functions and simulations.
def track_growth(myClass):
    '''
    
    This function plots the model functions and simulations.
    
    Input:
        myClass : Model class with parameters, grids, utility function, policy functions, and simulations.
        
    '''

    # Model parameters, policy and value functions, and simulations.
    par = myClass.par # Parameters.
    sol = myClass.sol # Policy functions.
    sim = myClass.sim # Simulations.

    # Production function.

    figure(1)
    plot(par.kgrid,sol.y)
    xlabel('$k_{t}$')
    ylabel('$y_{t}$') 
    title('Production Function')
    
    figname = myClass.par.figout+"\\ypol.png"
    print(figname)
    savefig(figname)

    # Plot capital policy function.
    
    figure(2)
    plot(par.kgrid,sol.k)
    xlabel('$k_{t}$')
    ylabel('$k_{t+1}$') 
    title('Capital Policy Function')
    
    figname = myClass.par.figout+"\\kpol.png"
    savefig(figname)

    # Plot consumption policy function.
    
    figure(3)
    plot(par.kgrid,sol.c)
    xlabel('$k_{t}$')
    ylabel('$c_{t}$') 
    title('Consumption Policy Function')
    
    figname = myClass.par.figout+"\\cpol.png"
    savefig(figname)

    # Plot investment policy function.
    
    figure(4)
    plot(par.kgrid,sol.i)
    xlabel('$k_{t}$')
    ylabel('$i_{t}$') 
    title('Investment Policy Function')
    
    figname = myClass.par.figout+"\\ipol.png"
    savefig(figname)

    # Plot value function.
    
    figure(5)
    plot(par.kgrid,sol.v)
    xlabel('$k_{t}$')
    ylabel('$V_t(k_t)$') 
    title('Value Function')

    figname = myClass.par.figout+"\\vfun.png"
    savefig(figname)
    
    # Plot simulated output.
    
    tgrid = linspace(1,par.T,par.T,dtype=int)

    figure(6)
    plot(tgrid,sim.ysim)
    xlabel('Time')
    ylabel('$y^{sim}_t$') 
    title('Simulated Output')

    figname = myClass.par.figout+"\\ysim.png"
    savefig(figname)
    
    # Plot simulated capital choice.
    
    figure(7)
    plot(tgrid,sim.ksim)
    xlabel('Time')
    ylabel('$k^{sim}_{t+1}$') 
    title('Simulated Capital Choice')

    figname = myClass.par.figout+"\\ksim.png"
    savefig(figname)
    
    # Plot simulated consumption.
    
    figure(8)
    plot(tgrid,sim.csim)
    xlabel('Time')
    ylabel('$c^{sim}_{t}$') 
    title('Simulated Consumption')

    figname = myClass.par.figout+"\\csim.png"
    savefig(figname)
    
    # Plot simulated investment.
    
    figure(9)
    plot(tgrid,sim.isim)
    xlabel('Time')
    ylabel('$i^{sim}_{t}$') 
    title('Simulated Investment')

    figname = myClass.par.figout+"\\isim.png"
    savefig(figname)
    
    # Plot simulated utility.
    
    figure(10)
    plot(tgrid,sim.usim)
    xlabel('Time')
    ylabel('$u^{sim}_t$') 
    title('Simulated Utility')

    figname = myClass.par.figout+"\\usim.png"
    savefig(figname)

    # Plot simulated productivity.
    
    figure(11)
    plot(tgrid,sim.Asim)
    xlabel('Time')
    ylabel('$A^{sim}_t$') 
    title('Simulated Productivity')

    # Plot simulated trade openness.

    figure(12)
    tgrid = linspace(1, par.T, par.T, dtype=int)
    plot(tgrid, sim.Tsim)
    xlabel("Time")
    ylabel("Trade Openness (T_t)")
    title("Simulated Trade Openness")
    figname = myClass.par.figout + "\\Tsim.png"
    savefig(figname)
    
    #show()
    #close('all')