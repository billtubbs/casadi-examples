from sympy import symbols, Symbol, Function, Eq, dsolve

x1, x2 = symbols('x1, x2', cls=Function)

A = Symbol('A', positive=True, nonzero=True)
t, x1_init, x2_init = symbols('t, x1_init, x2_init')
u1, u2, u3 = symbols('u1, u2, u3', positive=True)

ode_system = [
    Eq(x1(t).diff(t), (u1 - u3)  /  A), 
    Eq(x2(t).diff(t), u1 * u2 - u3 * x2(t)  /  (x1(t) * A))
]
ics = {x1(0): x1_init, x2(0): x2_init}

sol = dsolve(ode_system, [x1(t), x2(t)], ics=ics)

# Solution:
#
# In [28]: sol[0]
# Out[28]: Eq(x1(t), x1_init + t*(u1 - u3)/A)
# 
# In [29]: sol[1]
# Out[29]: Eq(x2(t), u2*(A*x1_init + t*u1 - t*u3) + (A**(u3/(u1 - u3))*x1_init**(u3/(u1 - u3))*x2_init - u2*exp(u1*log(A*C1)/(u1 - u3)))*exp(-u3*log(A*x1_init + t*u1 - t*u3)/(u1 - u3)))
#
# By experimentation I found that C1 = x1_init
#

def ivp_solution(t, x1_init, x2_init, u1, u2, u3, A):
    return [
        x1_init + t*(u1 - u3)/A,
        u2*(A*x1_init + t*u1 - t*u3) + (A**(u3/(u1 - u3))*x1_init**(u3/(u1 - u3))*x2_init - u2*np.exp(u1*np.log(A*x1_init)/(u1 - u3)))*np.exp(-u3*np.log(A*x1_init + t*u1 - t*u3)/(u1 - u3))
    ]

def ivp_solution2(t, x1_init, x2_init, u1, u2, u3, A):
    u1mu3 = u1 - u3
    Ax1initptu1mu3 = A * x1_init + t * u1mu3
    return [
        Ax1initptu1mu3 / A,
        u2 * Ax1initptu1mu3 + (A ** (u3 / u1mu3) * x1_init ** (u3 / u1mu3) * x2_init - u2 * np.exp(u1 * np.log(A * x1_init) / u1mu3)) * np.exp(-u3 * np.log(Ax1initptu1mu3) / u1mu3)
    ]

# Solve for the case when u1 == u3
ode_system2 = [
    Eq(x1(t).diff(t), 0), 
    Eq(x2(t).diff(t), u1 * (u2 - x2(t))  /  (x1(t) * A))
]

ics = {x1(0): x1_init, x2(0): x2_init}

sol2 = dsolve(ode_system2, [x1(t), x2(t)], ics=ics)