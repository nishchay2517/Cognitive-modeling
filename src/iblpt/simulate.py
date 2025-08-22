import numpy as np
from iblpt.agents.ibl import IBLAgent
from iblpt.agents.pt_ibl import PTIBLAgent


def eval_ts(dataset, params, use_pt=False, N=100, agents=5, rng_seed: int | None = None):
    """
    Evaluate R-rate and alternation-rate over N trials averaged across problems and agents.

    Returns (r_ts, a_ts) each length N.
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    P = len(dataset)
    r_acc = np.zeros(N)
    a_acc = np.zeros(N)
    for _, row in dataset.iterrows():
        R = np.zeros((agents, N))
        A = np.zeros((agents, N))
        for ai in range(agents):
            agent = PTIBLAgent(*params) if use_pt else IBLAgent(*params)
            seq   = agent.run_n(row, N)
            R[ai] = [int(c == 'risky') for c in seq]
            A[ai,0] = 0
            for t in range(1, N):
                A[ai,t] = int(seq[t] != seq[t-1])
        r_acc += R.mean(axis=0)
        a_acc += A.mean(axis=0)
    return r_acc / P, a_acc / P


