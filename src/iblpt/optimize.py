from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution
from functools import partial

from iblpt.simulate import eval_ts
from iblpt.metrics import msd
from iblpt.human.metrics import human_r_ts_est, human_a_ts_est


def _ibl_objective(x, dataset, N, agents, r_weight):
    """Objective function for IBL optimization."""
    r_ts, a_ts = eval_ts(dataset, x, False, N=N, agents=agents)
    return msd(r_ts, human_r_ts_est)*r_weight + msd(a_ts, human_a_ts_est)*(1-r_weight)


def _pt_objective(x, dataset, N, agents, r_weight):
    """Objective function for PT+IBL optimization."""
    r_ts, a_ts = eval_ts(dataset, x, True, N=N, agents=agents)
    return msd(r_ts, human_r_ts_est)*r_weight + msd(a_ts, human_a_ts_est)*(1-r_weight)


def fit_ibl(params0_bounds, dataset, N=100, agents=5, r_weight=0.9, callback=None, workers=1):
    """Fit IBL model parameters using differential evolution."""
    objective = partial(_ibl_objective, dataset=dataset, N=N, agents=agents, r_weight=r_weight)
    
    res = differential_evolution(
        objective, params0_bounds, maxiter=15, popsize=8,
        updating='deferred', workers=workers, tol=1e-2, callback=callback
    )
    return res


def fit_pt(params0_bounds, dataset, N=100, agents=5, r_weight=0.5, callback=None, workers=1):
    """Fit PT+IBL model parameters using differential evolution."""
    objective = partial(_pt_objective, dataset=dataset, N=N, agents=agents, r_weight=r_weight)
    
    res = differential_evolution(
        objective, params0_bounds, maxiter=15, popsize=8,
        updating='deferred', workers=workers, tol=1e-2, callback=callback
    )
    return res


# Callback functions to monitor the optimization
def create_callback(agent_type):
    """Create a callback function for monitoring optimization progress."""
    counter = 0
    def callback(xk, convergence=None):
        nonlocal counter
        counter += 1
        if agent_type == "IBL":
            print(f"[IBL] Gen {counter:2d}, params={xk}")
        else:
            print(f"[PT ] Gen {counter:2d}, params={xk}")
        return False
    return callback



