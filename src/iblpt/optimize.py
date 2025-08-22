from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution

from iblpt.simulate import eval_ts
from iblpt.metrics import msd
from iblpt.human.metrics import human_r_ts_est, human_a_ts_est


def fit_ibl(params0_bounds, dataset, N=100, agents=5, r_weight=0.9, callback=None):
    def objective(x):
        r_ts, a_ts = eval_ts(dataset, x, False, N=N, agents=agents)
        return msd(r_ts, human_r_ts_est)*r_weight + msd(a_ts, human_a_ts_est)*(1-r_weight)

    res = differential_evolution(
        objective, params0_bounds, maxiter=15, popsize=8,
        updating='deferred', workers=-1, tol=1e-2, callback=callback
    )
    return res


def fit_pt(params0_bounds, dataset, N=100, agents=5, r_weight=0.5, callback=None):
    def objective(x):
        r_ts, a_ts = eval_ts(dataset, x, True, N=N, agents=agents)
        return msd(r_ts, human_r_ts_est)*r_weight + msd(a_ts, human_a_ts_est)*(1-r_weight)

    res = differential_evolution(
        objective, params0_bounds, maxiter=15, popsize=8,
        updating='deferred', workers=-1, tol=1e-2, callback=callback
    )
    return res


