import opengen as og
import casadi.casadi as cs

# model constants
M = 1 # mass [kg]

# solver constants
DT = 0.1
N = 10 # horizon length
NX = 2 # number of state variables
NU = 1 # number of inputs

UMIN = [-1., ] * N
UMAX = [1.,] * N

Q = cs.diag(cs.SX([1., 1.]))
R = 1.

V = cs.diag(cs.SX([10., 10.]))

def system_ode(x: cs.SX, u: cs.SX) -> cs.SX:
    """the ODE defining the system
    x[0] -> position [m]
    x[1] -> velocity [m/s]
    u    -> force [N]
    """
    # sub into x for dx/dt
    x[0] = x[1]
    x[1] = u / M
    return x

def update_system(x: cs.SX, u: cs.SX) -> cs.SX:
    return x + DT * system_ode(x, u)

def stage_cost(x, u, x_ref):
    return cs.bilin(Q, (x - x_ref)) + R * u ** 2

def terminal_cost(x, x_ref):
    return cs.bilin(V, (x - x_ref))

def main():
    u_seq = cs.SX.sym("u", N * NU)  # sequence of all u's
    p = cs.SX.sym("p", 2 * NX)     # initial state, target state
    x, x_ref  = p[:NX], p[NX:]

    total_cost = 0

    for i in range(N):
        u = u_seq[i]

        total_cost += stage_cost(x, u, x_ref)  # update cost
        x = update_system(x, u)         # update state

    total_cost += terminal_cost(x, x_ref)  # terminal cost
    bounds = og.constraints.Rectangle(UMIN, UMAX)

    problem = og.builder.Problem(u_seq, p, total_cost)  \
        .with_constraints(bounds)

    build_config = og.config.BuildConfiguration()  \
        .with_build_directory("./solvers")      \
        .with_build_mode("debug")               \
        .with_build_python_bindings()

    meta = og.config.OptimizerMeta()  \
        .with_optimizer_name("linear_cart")

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