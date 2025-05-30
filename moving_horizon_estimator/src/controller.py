import opengen as og
import casadi.casadi as cs

from src.system_utils import system_ode, NX

# solver constants
DT = 0.1 # solver time step
N = 10 # horizon length
NU = 1 # number of inputs

Q = cs.diag(cs.SX([1.0, 1.0]))
R = 1.0

V = cs.diag(cs.SX([10.0, 10.0]))

def update_system(x: cs.SX, u: cs.SX) -> cs.SX:
    return x + DT * system_ode(x, u)

def stage_cost(x, u):
    return cs.bilin(Q, x) + R * u ** 2

def terminal_cost(x):
    return cs.bilin(V, x)

def main():
    u_seq = cs.SX.sym("u", N * NU)  # sequence of all u's
    p = cs.SX.sym("p", 2 * NX)     # initial state, target state
    x, x_ref  = p[:NX], p[NX:]

    total_cost = 0.0

    for i in range(N):
        u = u_seq[i]

        total_cost += stage_cost(x - x_ref, u)  # update cost
        x = update_system(x, u)         # update state

    total_cost += terminal_cost(x - x_ref)  # terminal cost
    bounds = og.constraints.BallInf(None, 1.0)

    problem = og.builder.Problem(u_seq, p, total_cost)  \
        .with_constraints(bounds)

    build_config = og.config.BuildConfiguration()  \
        .with_build_directory("./optimisers")      \
        .with_build_mode("debug")               \
        .with_build_python_bindings()

    meta = og.config.OptimizerMeta()  \
        .with_optimizer_name("controller")

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