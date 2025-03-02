def recurse(opti, intg, delta_num, N, x, x1_bound, X_history, U):
    # Builds a scenario tree recursively
    #
    # Settings to be passed as is: opti, intg, delta_num
    #
    # N: remaining horizon (shrinks while recursing)
    # x: previous state  (remains same size)
    # x1_bound: remaining list of bounds for state (shrinks while recursing)
    # X_history: a list with the history of the current full branch (grows while recursing)
    # U: remaining control variables (shrinks while recursing)
    #
    # This function return a list with an entry for each possible series of events (=one full branch)
    # Each list entry has a concatenation of all states of that full branch.
    import casadi as cas

    if N == 0:
        return [cas.hcat(X_history)]

    X = []
    for delta in [-delta_num, delta_num]:
        x_next = opti.variable(2)
        res = intg(x0=x, p=cas.vertcat(U[0], delta))
        opti.subject_to(x_next == res["xf"])
        
        opti.subject_to(opti.bounded(-0.25, x_next[0], x1_bound[0]))
        
        X_local = recurse(opti, intg, delta_num, N-1, x_next, x1_bound[1:], X_history+[x_next], U[1:])
        X += X_local
    return X
