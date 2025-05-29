import sys
import os

import numpy as np
import matplotlib.pyplot as plt

solver_dir = "solvers"
solver_name = "linear_cart"
sys.path.insert(1, os.path.join(solver_dir, solver_name))
linear_cart_mpc = __import__(solver_name)

from src.linear_cart import DT, NX, M, N, M, system_ode

# constants
T = 10

def update_system(x, u):
    dx_dt = system_ode(x, u)
    return [x[i] + DT * dx_dt[i] for i in range(NX)]

def plot_state_evolution(t_series, x_series, u_series, x_ref):
    """a function to plot the evolution of the system over time"""
    fig, ax = plt.subplots()

    # plot state first
    pass

def main():
    t_series = np.arange(0, T, DT)
    x_series = []
    u_series = []

    x = [0.0, 0.0]
    x_ref = [1.0, 0.0]

    solver = linear_cart_mpc.solver()

    for i, t in enumerate(t_series):
        x_series.append(x)
        # get control
        p =[x[0], x[1], x_ref[0], x_ref[1]]
        result = solver.run(p=p,)
        u_star = result.solution[0]
        u_series.append(u_star) 

        x = update_system(x, u_star)
        
    x_series = np.array(x_series)

    plt.step(t_series, u_series, where="post", color="r", linestyle="--")
    plt.plot(t_series, x_series)
    plt.show()

if __name__ == "__main__":
    main()

    


    