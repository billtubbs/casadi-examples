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

# Discrete-time approximation (Runge-Kutta method)
intg_options = {'number_of_finite_elements': 1}
solver = 'rk'
dae = {'x': x, 'p': u, 'ode': rhs}
t0 = 0
dt = 0.1
tf = dt / intg_options['number_of_finite_elements']
intg = cas.integrator('intg', solver, dae, t0, tf, intg_options)
res = intg(x0=x, p=u)
xkp1 = res['xf']
F = Function('F', [x, u], [xkp1], ['x', 'u'], ['xkp1'])
print(F)

# Simulation time
N = 25
t = np.arange(N+1) * float(dt)
T = t[-1]

opti = Opti()

# Decision variables for control action
U = opti.variable(nu, N)  # force [N]

# Decision variables for states
X = opti.variable(nx, N+1)

# Dynamic model constraints (multiple shooting)
for k in range(N):
    u = U[:, k]
    opti.subject_to(X[:, k+1] == F(X[:, k], U[k]))

# Define objective function
f = cas.sumsqr(U)
opti.minimize(f)

# Boundary constraints - initial condition
x0 = DM([15.0, 0.0])
opti.subject_to(X[0, 0] == x0[0])
opti.subject_to(X[1, 0] == x0[1])

# Boundary constraints - terminal condition
opti.subject_to(X[0, -1] == 0.0)
opti.subject_to(X[1, -1] > -0.01)

# Path constraint
opti.subject_to(X[0, :-1] > 0.0)
opti.subject_to(opti.bounded(0.0, U, 20.0))

opti.solver('ipopt')
sol = opti.solve()

xsol = np.array(sol.value(X)).T
# Add final value
usol = np.concatenate([sol.value(U), [np.nan]])

filename = "lander_ioplot.pdf"
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
    np.load(os.path.join(data_dir, "lander_ms_rk.npy")),
    equal_nan=True,
    atol=1e-15
)