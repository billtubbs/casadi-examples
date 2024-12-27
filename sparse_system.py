from pylab import *
from casadi import *

T = 1.0 # control horizon [s]
N = 40 # Number of control intervals

dt = T/N # length of 1 control interval [s]

##
# ----------------------------------
#    continuous system dot(x)=f(x,u)
# ----------------------------------

# Construct a CasADi function for the ODE right-hand side

A = np.array(
    [[0,0,0,0,0,0,1,0],
     [0,0,0,0,0,1,0,1],
     [0,0,0,0,1,0,0,1],
     [0,0,0,1,0,0,0,1],
     [0,0,1,1,1,1,1,1],
     [0,1,0,0,0,0,0,1],
     [1,0,0,0,0,0,0,1],
     [1,1,1,0,0,0,1,1]])
nx = A.shape[0]
B = np.array(
 [[-0.073463,-0.073463],
  [-0.146834,-0.146834],
  [-0.146834,-0.146834],
  [-0.146834,-0.146834],
  [-0.446652,-0.446652],
  [-0.147491,-0.147491],
  [-0.147491,-0.147491],
  [-0.371676,-0.371676]])

nu = B.shape[1]

x  = MX.sym('x',nx)
u  = MX.sym('u',nu)

dx = sparsify(A) @ sqrt(x)+sparsify(B) @ u

x_steady = (-solve(A,B @ vertcat(1,1)))**2

# Continuous system dynamics as a CasADi Function
f = Function('f', [x, u], [dx])

# -----------------------------------
#    Discrete system x_next = F(x,u)
# -----------------------------------

# Reference Runge-Kutta implementation
opts = dict()
opts["tf"] = dt
opts["number_of_finite_elements"] = 1
intg = integrator('intg','rk',{'x':x,'p':u,'ode':f(x,u)},opts)

##
# -----------------------------------------------
#    Optimal control problem, multiple shooting
# -----------------------------------------------

opti = casadi.Opti()

# Decision variables for states
X = opti.variable(nx,N+1)
# Decision variables for control vector
U = opti.variable(nu,N)

# Gap-closing shooting constraints
for k in range(N):
   res = intg(x0=X[:,k],p=U[:,k])
   opti.subject_to(X[:,k+1]==res["xf"])

# Path constraints
opti.subject_to(opti.bounded(0.01,vec(X),0.1))

# Initial guesses
opti.set_initial(X, repmat(x_steady,1,N+1))
opti.set_initial(U, 1)

# Initial and terminal constraints
opti.subject_to(X[:,0]==x_steady);
# Objective: regularization of controls

xbar = opti.variable();
opti.minimize(1e-6*sumsqr(U)+sumsqr(X[:,-1]-xbar))

# solve optimization problem
opti.solver('ipopt')

sol = opti.solve()

##
# -----------------------------------------------
#    Post-processing: plotting
# -----------------------------------------------

figure()
Xsol = sol.value(X)
plot(Xsol.T,'o-')

show()
