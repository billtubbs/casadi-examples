import os
import numpy as np
import matplotlib.pyplot as plt
import casadi as cas
from casadi import DM, MX, Function, Opti
from plot_utils import make_uxplot


data_dir = 'data'
plot_dir = 'plots'
os.makedirs(plot_dir, exist_ok=True)

# ODE right-hand side
#
# dx[0]/dt = x[1]
# dx[1]/dt = u
#
nx = 2
nu = 1
ny = 1
m = 1.0  # mass
g = -9.81  # acceleration due to gravity
x = MX.sym('x', nx)  # states: pos [m], vel [m/s]
u = MX.sym('u', nu)  # control force [N]
rhs = cas.vertcat(x[1], u / m + g)
f = Function('f', [x, u], [rhs], ['x', 'u'], ['rhs'])

# Discrete-time approximation (explicit Euler method)
dt = DM(0.1)
xkp1 = x + dt * f(x, u)
F = Function('F', [x, u], [xkp1], ['x', 'u'], ['xkp1'])
print(F)

# Simulation time
N = 25
t = np.arange(N+1) * float(dt)
T = t[-1]

opti = Opti()

# Decision variables for control action
U = opti.variable(nu, N)  # force [N]

# Trajectory simulation (single shooting)
x0 = MX.sym('x0', 2)
X = []
xk = x0
for k in range(N):
    u = U[:, k]
    xk = F(xk, u)
    X.append(xk)
X = cas.hcat(X)

simulate = Function('simulate', [x0, U], [X], ['x0', 'U'], ['X'])
print(simulate)

x0 = DM([15, 0])
X = simulate(x0, U)

# Boundary constraints
opti.subject_to(X[0, -1] == 0.0)
opti.subject_to(X[1, -1] > -0.01)

# Path constraint
opti.subject_to(X[0, :-1] > 0.0)
opti.subject_to(opti.bounded(0.0, U, 20.0))
opti.minimize(cas.sumsqr(U))

opti.solver('ipopt')
sol = opti.solve()

# Add initial and final values
xsol = np.concatenate([x0, sol.value(X)], axis=1).T
usol = np.concatenate([sol.value(U), [np.nan]])

filename = "lander_ss_ioplot.pdf"
x_titles = ['Position', 'Velocity']
u_titles = ['Thrust']
fig, axes = make_uxplot(
    t, 
    usol.reshape(-1, nu), 
    xsol, 
    x_titles=x_titles, 
    u_titles=u_titles, 
    filename=filename, 
    plot_dir=plot_dir
)

print("\nClose plot window to end script.")
plt.show()

# Test results match data on file
assert np.allclose(
    np.hstack([xsol, usol.reshape(-1, 1)]),
    np.load(os.path.join(data_dir, "lander_ss.npy")),
    equal_nan=True,
    atol=1e-15
)
