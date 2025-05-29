import sys
import os

import numpy as np
import matplotlib.pyplot as plt

solver_dir = "solvers"
solver_name = "linear_cart"
sys.path.insert(1, os.path.join(solver_dir, solver_name))
linear_cart_mpc = __import__(solver_name)

from src.linear_cart import DT, NX, M, N

# constants
T = 60

def update_system(x, u):
    x[0] += DT * x[1]
    x[1] += DT * u / M

    return x

def plot_state_evolution(t_series, x_series, u_series, x_ref):
    """a function to plot the evolution of the system over time"""
    fig, ax = plt.subplots()

    # plot state first
    pass

def main():
    t_series = np.arange(0, T, DT, dtype=float)
    x_series = np.empty((len(t_series), NX), dtype=float)
    u_series = np.empty(len(t_series), dtype=float)

    x = np.array([0.0, 0.0], dtype=float)
    x_ref = np.array([10.0, 0.0], dtype=float)

    solver = linear_cart_mpc.solver()
    u_prev = np.zeros(N)

    for i, t in enumerate(t_series):
        x_series[i] = x.copy()
        # get control
        p =[x[0], x[1], x_ref[0], x_ref[1]]
        result = solver.run(p=p, initial_guess=u_prev)
        u_prev = result.solution
        u_star = u_prev[0]
        x = update_system(x, u_star)
        
        u_series[i] = u_star 

    plt.plot(t_series, u_series, "r--")
    plt.plot(t_series, x_series)
    # plt.show()

if __name__ == "__main__":
    main()

    


    