import sys
import os

import numpy as np
import matplotlib.pyplot as plt

optimiser_dir = "optimisers"
controller = "controller"
estimator = "estimator"
sys.path.insert(1, os.path.join(optimiser_dir, controller))
linear_cart_mpc = __import__(controller)
sys.path.insert(1, os.path.join(optimiser_dir, estimator))
linear_cart_estimator = __import__(estimator)

from src.controller import DT, N
from src.estimator import measure_system
from src.system_utils import system_ode, NX

# seed random
np.random.seed(42)

# constants
T = 20
NOISE = 1e-2

def update_system(x, u):
    dx_dt = system_ode(x, u)
    return [x[i] + DT * dx_dt[i] for i in range(NX)]

def plot_state_evolution(t_series, x_series, u_series, x_ref):
    """a function to plot the evolution of the system over time"""
    fig, ax = plt.subplots()

    # plot state first
    pass

def main():
    """this example aims to control a linear cart, while only being able
    to measure the position of the cart. Additionally this measurement is
    subject to random noise, so a moving horizon estimator is used"""
    t_series = np.arange(0, T, DT)
    x_series = []
    x_hat_series = []
    y_series = []
    u_series = []

    y_horizon = [0,] * N
    u_horizon = [0,] * N

    x = [0.0, 0.0]
    x_hat = [0.0, 0.0]
    x_ref = [1.0, 0.0]

    controller = linear_cart_mpc.solver()
    estimator = linear_cart_estimator.solver()

    for i, t in enumerate(t_series):
        x_series.append(x) # record true x 
        x_hat_series.append(x_hat)

        # get control based on state estimate and setpoint
        p =[x_hat[0], x_hat[1], x_ref[0], x_ref[1]]
        result = controller.run(p=p,)
        u_star = result.solution[0] # control for the estimated state

        u_series.append(u_star) 

        # update guess horizons
        x = update_system(x, u_star + np.random.normal() * NOISE) # uncertain input
        y = measure_system(x) + np.random.normal() * NOISE # noise on measurement 

        # extend estimator horizons
        y_horizon.append(y)
        y_horizon.pop(0)
        u_horizon.append(u_star)
        u_horizon.pop(0)

        # call estimator to estimate next x_hat
        estimator_result = estimator.run(p=[*y_horizon, *u_horizon])
        x_hat = estimator_result.solution[-NX:]

        
    x_series = np.array(x_series)
    x_hat_series = np.array(x_hat_series)

    plt.step(t_series, u_series, where="post", color="r", linestyle="--")
    plt.plot(t_series, x_series, "-")
    plt.plot(t_series, x_hat_series, "--")
    plt.show()

if __name__ == "__main__":
    main()

    


    