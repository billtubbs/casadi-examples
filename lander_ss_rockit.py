import os
import matplotlib.pyplot as plt
import numpy as np
import rockit
from rockit import SingleShooting, MultipleShooting


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

# Simulation time
N = 25
dt = 0.1
t = np.arange(N+1) * float(dt)
T = t[-1]

# Optimal control problem
ocp = rockit.Ocp(T=T)
x1 = ocp.state()
x2 = ocp.state()
u = ocp.control()
ocp.set_der(x1, x2)
ocp.set_der(x2, u / m + g)

# Boundary constraints
x1_0, x2_0 = 15, 0
ocp.subject_to(ocp.at_tf(x1) == 0.0)
ocp.subject_to(ocp.at_tf(x2) > -0.01)
ocp.subject_to(ocp.at_t0(x1) == x1_0)
ocp.subject_to(ocp.at_t0(x2) == x2_0)

# Path constraint
ocp.subject_to(x1 >= 0.0)
ocp.subject_to(0 <= (u <= 20 ))
ocp.add_objective(ocp.integral(u**2))

ocp.solver('ipopt')

method = SingleShooting(N=N, intg='expl_euler')
ocp.method(method)

sol = ocp.solve()
t1, x1_sol = sol.sample(x1, grid='control')
assert np.array_equal(t, t1)
t1, x2_sol = sol.sample(x2, grid='control')
assert np.array_equal(t, t1)
t1, u_sol = sol.sample(u, grid='control')
assert np.array_equal(t, t1)

fig, axes = plt.subplots(3, 1, sharex=True)

ax = axes[0]
ax.plot(t, x1_sol, '.-')
ax.grid()
ax.set_ylabel('$x_1$')
ax.set_title('Position')

ax = axes[1]
ax.plot(t, x2_sol, '.-')
ax.grid()
ax.set_ylabel('$x_2$')
ax.set_title('Velocity')

ax = axes[2]
ax.plot(t, u_sol, '.-')
ax.grid()
ax.set_xlabel('Time')
ax.set_ylabel('$u$')
ax.set_title('Thrust')

plt.tight_layout()
filename = "lander_rockit_ioplot.pdf"
plt.savefig(os.path.join(plot_dir, filename))
print("\nClose plot window to end script.")
plt.show()

# Test results match data on file
u_sol[-1] = np.nan  # needed for comparison with CasADi solution in lander_ss.py
assert np.allclose(
    np.stack([x1_sol, x2_sol, u_sol]).T,
    np.load(os.path.join(data_dir, "lander_ss.npy")),
    equal_nan=True,
    atol=1e-7  # TODO: Should it be less?
)
