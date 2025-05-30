import opengen as og
import casadi.casadi as cs

from src.system_utils import system_ode, NX
from src.controller import DT, NU

# estimator constants
N = 10 # horizon length
NY = 1

Q = cs.diag(cs.SX([20.0, 20.0]))
R = 1.0

def update_system(x: cs.SX, u: cs.SX) -> cs.SX:
    return x + DT * system_ode(x, u)

def measure_system(x):
    """return the system measurement, y"""
    return x[0] # return only position

def state_noise(x1, x0, u0):
    return cs.bilin(Q, x1 - update_system(x0, u0))

def measurement_noise(x, y):
    return R * (y - measure_system(x)) ** 2

def main():
    # NOTE: the final x in this seq is the estimate of the current state
    x_hat_seq = cs.SX.sym("x_hat", (N + 1) * NX)  # sequence of all estimated states
    p = cs.SX.sym("p", N * (NY + NU)) # N * y, N * u
    y_seq, u_seq  = p[: N * NY], p[N * NY:]

    total_cost = 0.0

    for i in range(N):
        u = u_seq[i]
        y = y_seq[i]
        x_hat_0 = x_hat_seq[i * NX : (i + 1) * NX]
        x_hat_1 = x_hat_seq[(i + 1) * NX : (i + 2) * NX]

        total_cost += state_noise(x_hat_1, x_hat_0, u)
        total_cost += measurement_noise(x_hat_0, y)

    problem = og.builder.Problem(x_hat_seq, p, total_cost)

    build_config = og.config.BuildConfiguration()  \
        .with_build_directory("./optimisers")      \
        .with_build_mode("debug")               \
        .with_build_python_bindings()

    meta = og.config.OptimizerMeta()  \
        .with_optimizer_name("estimator")

    solver_config = og.config.SolverConfiguration()\
        .with_tolerance(1e-6) \
        .with_initial_tolerance(1e-6) \
        .with_max_outer_iterations(1000) \
        .with_initial_penalty(1)

    builder = og.builder.OpEnOptimizerBuilder(problem, meta,
                                            build_config, solver_config)
    builder.build()

if __name__ == "__main__":
    main()