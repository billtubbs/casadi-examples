import numpy as np
from casadi import DM, MX, Function
import casadi as cas
from lagrange_polynomial_eval import LagrangePolynomialEval


def sparse_system():
    """Construct a CasADi function for the ODE right-hand side of
    the continuous system model:

        dot(x) = f(x, u)

    """

    A = np.array(
        [[0, 0, 0, 0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 1, 0, 1], 
        [0, 0, 0, 0, 1, 0, 0, 1], 
        [0, 0, 0, 1, 0, 0, 0, 1], 
        [0, 0, 1, 1, 1, 1, 1, 1], 
        [0, 1, 0, 0, 0, 0, 0, 1], 
        [1, 0, 0, 0, 0, 0, 0, 1], 
        [1, 1, 1, 0, 0, 0, 1, 1]])

    B = np.array(
    [[-0.073463, -0.073463], 
    [-0.146834, -0.146834], 
    [-0.146834, -0.146834], 
    [-0.146834, -0.146834], 
    [-0.446652, -0.446652], 
    [-0.147491, -0.147491], 
    [-0.147491, -0.147491], 
    [-0.371676, -0.371676]])

    nx = A.shape[0]
    nu = B.shape[1]
    x  = MX.sym('x', nx)
    u  = MX.sym('u', nu)

    dx = cas.sparsify(A) @ cas.sqrt(x) + cas.sparsify(B) @ u

    # Continuous system dynamics as a CasADi Function
    f = Function('f', [x, u], [dx])

    # Steady-state values of states
    x_steady = (-cas.solve(A, B @ cas.vertcat(1, 1))) ** 2

    return x, u, f, nx, nu, x_steady


def cas_integrator(x, u, f, dt, solver='rk', number_of_finite_elements=1):
    """Construct a CasADi integrator for the discrete system
    transition function:

        x_next = F(x, u)

    """
    dae = {'x': x, 'p': u, 'ode': f(x, u)}
    t0 = 0.0
    tf = dt
    opts = {"number_of_finite_elements": number_of_finite_elements}
    return cas.integrator('intg', solver, dae, t0, tf, opts)


def ocp_multiple_shooting(nx, nu, N, intg, x_steady, solver='ipopt'):
    """Set up Optimal control problem, multiple shooting
    """

    opti = cas.Opti()

    # Decision variables for states
    X = opti.variable(nx, N + 1)

    # Decision variables for control vector
    U = opti.variable(nu, N)

    # Gap-closing shooting constraints
    for k in range(N):
        res = intg(x0=X[:, k], p=U[:, k])
        opti.subject_to(X[:, k + 1] == res["xf"])

    # Path constraints
    opti.subject_to(opti.bounded(0.01, cas.vec(X), 0.1))

    # Initial guesses
    opti.set_initial(X, cas.repmat(x_steady, 1, N + 1))
    opti.set_initial(U, 1)

    # Initial and terminal constraints
    opti.subject_to(X[:, 0] == x_steady)
    # Objective: regularization of controls

    xbar = opti.variable()
    opti.minimize(1e-6 * cas.sumsqr(U) + cas.sumsqr(X[:, -1] - xbar))

    # solve optimization problem
    opti.solver(solver)

    return opti, X, U


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


def make_polynomial_eval_functions_alternate(dt, nx, d, t0=0.0, scheme='legendre'):

    # Define collocation times
    tau = DM(cas.collocation_points(d, scheme)).T
    t_coll = t0 + dt * tau
    T = cas.horzcat(t0, t_coll)

    # Define collocation points
    X0 = MX.sym("X0", nx)
    Xc = MX.sym("Xc", nx, d)
    X = cas.horzcat(X0, Xc)

    # Polynomial function
    t = MX.sym('t')
    Pi_expr = LagrangePolynomialEval(T, X, t)

    # Time derivative
    dPidt_expr = cas.jacobian(Pi_expr, t)

    Pi = Function('Pi', [t, X0, Xc], [Pi_expr], ['t', 'X0', 'Xc'], ['Pi'])
    dot_Pi = Function('dot_Pi', [t, X0, Xc], [dPidt_expr], ['t', 'X0', 'Xc'], ['dPidt'])

    return Pi, dot_Pi, t_coll


def ocp_direct_collocation(
        f, nx, nu, N, dt, d, tau, Pi, dot_Pi, x_steady, solver='ipopt'
    ):
    """Set up Optimal control problem, multiple shooting
    """

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
        t_coll = t0 + dt * DM(tau).T
        tf = (k + 1) * dt
        Xck = Xc[:, k*d:(k+1)*d]
        dxdt = dot_Pi(t0, t_coll, t_coll, X[:, k], Xck)
        opti.subject_to(dxdt == f(Xck, U[:, k]))

        # Continuity constraint
        xf = Pi(t0, t_coll, tf, X[:, k], Xck)
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
    opti.solver(solver)

    return opti, X, U


def ocp_direct_collocation_coeffs(
        f, nx, nu, N, dt, d, tau, x_steady, solver='ipopt'
    ):
    """Set up Optimal control problem, multiple shooting
    """

    # Linear maps to compute Pi and dot_Pi
    C, D, B = cas.collocation_coeff(tau)

    opti = cas.Opti()

    # Decision variables for states
    X = opti.variable(nx, N + 1)

    # Decision variables for control vector
    U = opti.variable(nu, N)

    # Gap-closing shooting constraints with collocation
    for k in range(N):

        # Decision variables for collocation scheme
        Xc = opti.variable(nx, d)

        Z = cas.horzcat(X[:, k], Xc)
        dot_Pi = (Z @ C) / dt
        opti.subject_to(dot_Pi == f(Xc, U[:, k]))

        # Continuity constraint
        Pi_f = Z @ D
        opti.subject_to(Pi_f == X[:, k + 1])
        opti.set_initial(Xc, cas.repmat(x_steady, 1, d))

    # Path constraints
    opti.subject_to(opti.bounded(0.01, cas.vec(X), 0.1))

    # Initial guesses
    opti.set_initial(X, cas.repmat(x_steady, 1, N + 1))

    opti.set_initial(U, 1)

    # Initial and terminal constraints
    opti.subject_to(X[:, 0] == x_steady)
    # Objective: regularization of controls

    xbar = opti.variable()
    opti.minimize(1e-6 * cas.sumsqr(U) + cas.sumsqr(X[:, -1] - xbar))

    # solve optimization problem
    opti.solver(solver)

    return opti, X, U


def process_timing(sol):
    stats = sol.stats()
    return {
        name: stats[name]
        for name in [
            't_proc_callback_fun',
            't_proc_nlp_f',
            't_proc_nlp_g',
            't_proc_nlp_grad',
            't_proc_nlp_grad_f',
            't_proc_nlp_hess_l',
            't_proc_nlp_jac_g',
            't_proc_total'
        ]
    }