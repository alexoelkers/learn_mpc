import casadi.casadi as cs

# model constants
M = 1.0 # mass [kg]
NX = 2

def system_ode(x: cs.SX, u: cs.SX) -> cs.SX:
    """the ODE defining the system
    x[0] -> position [m]
    x[1] -> velocity [m/s]
    u    -> force [N]
    """
    # sub into x for dx/dt
    dx_dt = cs.SX.sym("dx_dt", NX)
    dx_dt[0] = x[1]
    dx_dt[1] = u / M
    return dx_dt