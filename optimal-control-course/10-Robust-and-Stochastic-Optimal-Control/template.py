import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
from casadi import MX, Opti

# Physical constants

T = 1.0 # control horizon [s]
N = 40 # Number of control intervals

dt = T/N # length of 1 control interval [s]

tgrid = np.linspace(0,T,N+1)

##
# ----------------------------------
#    continuous system dot(x)=f(x,u)
# ----------------------------------
nx = 2

# Construct a CasADi function for the ODE right-hand side
x1 = MX.sym('x1')
x2 = MX.sym('x2')
u = MX.sym('u') # control
rhs = cas.vertcat(x2, -0.1 * (1 - x1**2) * x2 - x1 + u)
x = cas.vertcat(x1, x2)

x1_bound = lambda t: 2 + 0.1 * cas.cos(10 * t)

##
# -----------------------------------
#    Discrete system x_next = F(x,u)
# -----------------------------------

opts = dict()
opts["tf"] = dt
intg = cas.integrator('intg', 'cvodes', {'x': x, 'p': u, 'ode': rhs}, opts)

##
# -----------------------------------------------
#    Optimal control problem, multiple shooting
# -----------------------------------------------
x0 = cas.vertcat(0.5,0)

opti = Opti()

# Decision variable for states
x = opti.variable(nx)

# Initial constraints
opti.subject_to(x==x0)

U = []
X = [x]
# Gap-closing shooting constraints
for k in range(N):
    u = opti.variable()
    U.append(u)

    x_next = opti.variable(nx)
    res = intg(x0=x, p=u)
    opti.subject_to(x_next == res["xf"])

    opti.subject_to(opti.bounded(-40, u, 40))
    opti.subject_to(opti.bounded(-0.25, x[0], x1_bound(tgrid[k])))

    x = x_next
    X.append(x)

opti.subject_to(opti.bounded(-0.25, x_next[0], x1_bound(tgrid[N])))
U = cas.hcat(U)
X = cas.hcat(X)

opti.minimize(cas.sumsqr(X[0,:]-3))

opti.solver('ipopt')

sol = opti.solve()

##
# -----------------------------------------------
#    Post-processing: plotting
# -----------------------------------------------


# Simulate forward in time using an initial state and control vector
usol = sol.value(U)
xsol = sol.value(X)

plt.figure()
plt.plot(tgrid,xsol[0,:].T,'bs-','linewidth',2)
plt.plot(tgrid,x1_bound(tgrid),'r--','linewidth',4)
plt.legend(('OCP trajectory x1','bound on x1'))
plt.xlabel('Time [s]')
plt.ylabel('x1')

plt.figure()
print(usol.shape)
plt.step(tgrid,cas.vertcat(usol,usol[-1]))
plt.title('applied control signal')
plt.ylabel('Force [N]')
plt.xlabel('Time [s]')

plt.show()
