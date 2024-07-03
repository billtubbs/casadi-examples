from itertools import product
from casadi import hcat


def recurse_dummy(delta_num, N):
    """Builds a scenario tree recursively
    
    Settings to be passed as is: delta_num
    
    N: remaining horizon (shrinks while recursing)
    history: a list with the history of the current full branch (grows while recursing)

    This function returns a list of all possible sequences of events (=one full branch)
    Each item has a concatenation of all disturbances acting in that full branch.\
    """

    delta_values = [[-delta_num, delta_num]] * N
    return [hcat(seq) for seq in product(*delta_values)]
