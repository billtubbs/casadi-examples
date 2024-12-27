import matplotlib.pyplot as plt
import numpy as np
from casadi import DM, MX, Function
import casadi as cas
from lagrange_polynomial_eval import LagrangePolynomialEval


T = 1.0  # control horizon [s]
N = 40  # Number of control intervals

dt = T / N  # length of 1 control interval [s]

##
# ----------------------------------
#    continuous system dot(x)=f(x, u)
# ----------------------------------

# Construct a CasADi function for the ODE right-hand side

A = np.array(
    [[0, 0, 0, 0, 0, 0, 1, 0], 
     [0, 0, 0, 0, 0, 1, 0, 1], 
     [0, 0, 0, 0, 1, 0, 0, 1], 
     [0, 0, 0, 1, 0, 0, 0, 1], 
     [0, 0, 1, 1, 1, 1, 1, 1], 
     [0, 1, 0, 0, 0, 0, 0, 1], 
     [1, 0, 0, 0, 0, 0, 0, 1], 
     [1, 1, 1, 0, 0, 0, 1, 1]])
nx = A.shape[0]
B = np.array(
 [[-0.073463, -0.073463], 
  [-0.146834, -0.146834], 
  [-0.146834, -0.146834], 
  [-0.146834, -0.146834], 
  [-0.446652, -0.446652], 
  [-0.147491, -0.147491], 
  [-0.147491, -0.147491], 
  [-0.371676, -0.371676]])

nu = B.shape[1]

x  = MX.sym('x', nx)
u  = MX.sym('u', nu)

dx = cas.sparsify(A) @ cas.sqrt(x) + cas.sparsify(B) @ u

x_steady = (-cas.solve(A, B @ cas.vertcat(1, 1))) ** 2

# Continuous system dynamics as a CasADi Function
f = Function('f', [x, u], [dx])

# -----------------------------------
#    Collocation scheme
# -----------------------------------

d = 3  # degree
scheme = 'radau'
tau = DM(cas.collocation_points(d, scheme)).T


def make_polynomial_functions(nx, d):
    t0 = MX.sym("t0")
    t_coll = MX.sym("t_coll", 1, d)
    T = cas.horzcat(t0, t_coll)
    X0 = MX.sym("X0", nx)
    Xc = MX.sym("Xc", nx, d)
    X = cas.horzcat(X0, Xc)

    t = MX.sym('t')
    Pi_expr = LagrangePolynomialEval(T, X, t)
    Pi = Function(
        'Pi', 
        [t0, t_coll, t, X0, Xc], 
        [Pi_expr], 
        ['t0', 't_coll', 't', 'X0', 'Xc'], 
        ['Pi']
    )

    dPidt_expr = cas.jacobian(Pi_expr, t)
    dot_Pi = Function(
        'dot_Pi', 
        [t0, t_coll, t, X0, Xc], 
        [dPidt_expr], 
        ['t0', 't_coll', 't', 'X0', 'Xc'], 
        ['dPidt']
    )

    return Pi, dot_Pi


# Define polynomial functions
Pi, dot_Pi = make_polynomial_functions(nx, d)

##
# -----------------------------------------------
#    Optimal control problem, multiple shooting
# -----------------------------------------------

opti = cas.Opti()

# Decision variables for states
X = opti.variable(nx, N + 1)
# Decision variables for control vector
U = opti.variable(nu, N)
# Decision variables for collocation scheme
Xc = opti.variable(nx, d * (N + 1))

# Gap-closing shooting constraints with collocation
for k in range(N):
    t0 = k * dt
    t_coll = t0 + dt * tau
    tf = (k + 1) * dt
    Xck = Xc[:, k*d:(k+1)*d]
    Uck = cas.repmat(U[:, k], 1, d)
    xf = Pi(t0, t_coll, tf, X[:, k], Xck)
    dxdt = dot_Pi(t0, t_coll, t_coll, X[:, k], Xck)
    opti.subject_to(dxdt == f(Xck, Uck))
    opti.subject_to(X[:, k + 1] == xf)

# Path constraints
opti.subject_to(opti.bounded(0.01, cas.vec(X), 0.1))

# Initial guesses
opti.set_initial(X, cas.repmat(x_steady, 1, N + 1))
opti.set_initial(Xc, cas.repmat(x_steady, 1, d * (N + 1)))
opti.set_initial(U, 1)

# Initial and terminal constraints
opti.subject_to(X[:, 0] == x_steady)
# Objective: regularization of controls

xbar = opti.variable()
opti.minimize(1e-6 * cas.sumsqr(U) + cas.sumsqr(X[:, -1] - xbar))

# solve optimization problem
opti.solver('ipopt')

sol = opti.solve()

##
# -----------------------------------------------
#    Post-processing: plotting
# -----------------------------------------------

plt.figure()
Xsol = sol.value(X)
plt.plot(Xsol.T, 'o-')

plt.show()
