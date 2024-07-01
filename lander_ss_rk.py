import casadi as cas
from casadi import DM, MX, Function, Opti
import matplotlib.pyplot as plt
import numpy as np
import os


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

# Decision variables for control
U = opti.variable(nu, N) # force [N]

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

fig, axes = plt.subplots(3, 1, sharex=True)

ax = axes[0]
ax.plot(t, xsol[:, 0], '.-')
ax.grid()
ax.set_ylabel('$x_1$')
ax.set_title('Position')

ax = axes[1]
ax.plot(t, xsol[:, 1], '.-')
ax.grid()
ax.set_ylabel('$x_2$')
ax.set_title('Velocity')

ax = axes[2]
ax.plot(t, usol, '.-')
ax.grid()
ax.set_xlabel('Time')
ax.set_ylabel('$u$')
ax.set_title('Thrust')

plt.tight_layout()
filename = "lander_ioplot.pdf"
plt.savefig(os.path.join(plot_dir, filename))
print("\nClose plot window to end script.")
plt.show()

# Test results match data on file
assert np.allclose(
    np.hstack([xsol, usol.reshape(-1, 1)]),
    np.load(os.path.join(data_dir, "lander_ss_rk.npy")),
    equal_nan=True,
    atol=1e-15
)