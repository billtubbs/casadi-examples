import numpy as np
import matplotlib.pyplot as plt
import casadi as cas
from casadi import MX, Function, Opti


# Physical constants
T = 1.0 # control horizon [s]
N = 20 # Number of control intervals
dt = T/N # length of 1 control interval [s]

tgrid = np.linspace(0, T, N+1)

##
# ----------------------------------
#    continuous system dot(x)=f(x,u)
# ----------------------------------
nx = 4

# Construct a CasADi function for the ODE right-hand side
x = MX.sym('x', nx) # states: pos_x [m], pos_y [m], vel_x [m/s], vel_y [m/s]
u = MX.sym('u', 2) # control force [N]
rhs = cas.vertcat(x[2:4], u)

# Continuous system dynamics as a CasADi Function
f = Function('f', [x, u], [rhs])


##
# -----------------------------------
#    Discrete system x_next = F(x,u)
# -----------------------------------

k1 = f(x, u)
k2 = f(x + dt/2 * k1, u)
k3 = f(x + dt/2 * k2, u)
k4 = f(x + dt * k3, u)
xf = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

F = Function('F', [x, u], [xf])

##
# ------------------------------------------------
# Waypoints
# ------------------------------------------------

ref = cas.horzcat(cas.sin(np.linspace(0, 2, N+1)), cas.cos(np.linspace(0, 2, N+1))).T

##
# -----------------------------------------------
#    Optimal control problem, multiple shooting
# -----------------------------------------------

opti = Opti()

# Decision variables for states
X = opti.variable(nx, N+1)

# Decision variables for control actions
U =  opti.variable(2, N) # force [N]

# Gap-clocas.sing shooting constraints
for k in range(N):
  opti.subject_to(X[:, k+1]==F(X[:, k], U[:, k]))

# Path constraints
opti.subject_to(opti.bounded(-3, X[0, :], 3)) # pos_x limits
opti.subject_to(opti.bounded(-3, X[1, :], 3)) # pos_y limits
opti.subject_to(opti.bounded(-3, X[2, :], 3)) # vel_x limits
opti.subject_to(opti.bounded(-3, X[3, :], 3)) # vel_y limits
opti.subject_to(opti.bounded(-10, U[0, :], 10)) # force_x limits
opti.subject_to(opti.bounded(-10, U[1, :], 10)) # force_x limits

# Initial constraints
opti.subject_to(X[:,0] == cas.vertcat(ref[:, 0], 0, 0))

# Try to follow the waypoints
opti.minimize(cas.sumsqr(X[:2, :] - ref))

opti.solver('ipopt')

sol = opti.solve()

##
# -----------------------------------------------
#    Post-processing: plotting
# -----------------------------------------------

xsol = sol.value(X)
usol = sol.value(U)

plt.figure()
plt.plot(xsol[0, :].T, xsol[1, :].T, 'bs-', linewidth=2)
plt.plot(ref[0, :].T, ref[1, :].T, 'ro', linewidth=3)
plt.legend(('OCP trajectory', 'Reference trajecory'))
plt.title('Top view')
plt.xlabel('x')
plt.xlabel('y')

plt.figure()
plt.step(tgrid,cas.horzcat(usol, usol[:, -1]).T)
plt.title('Applied control signal')
plt.legend(('force_x', 'force_y'))
plt.ylabel('Force [N]')
plt.xlabel('Time [s]')
plt.show()
