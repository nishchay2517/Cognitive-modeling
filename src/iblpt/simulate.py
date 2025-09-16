import numpy as np
import pandas as pd
from iblpt.agents.ibl import IBLAgent
from iblpt.agents.pt_ibl import PTIBLAgent


def eval_ts(dataset, params, use_pt=False, N=100, agents=5, rng_seed=None):
    """
    Evaluate R-rate and alternation-rate over N trials averaged across problems and agents.

    Args:
        dataset: pandas DataFrame with problem data
        params: tuple of parameters for the agent
        use_pt: whether to use PT+IBL agent (True) or IBL agent (False)
        N: number of trials
        agents: number of Monte Carlo replications per problem
        rng_seed: random seed for reproducibility

    Returns:
        tuple: (r_ts, a_ts) each length N
    """
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("dataset must be a pandas DataFrame")
    
    if len(params) < 3:
        raise ValueError("params must have at least 3 values")
    
    if use_pt and len(params) < 6:
        raise ValueError("PT+IBL agent requires 6 parameters")
    
    if N <= 0:
        raise ValueError("N must be positive")
    
    if agents <= 0:
        raise ValueError("agents must be positive")
    
    if rng_seed is not None:
        np.random.seed(rng_seed)

    P = len(dataset)
    if P == 0:
        raise ValueError("dataset is empty")
    
    r_acc = np.zeros(N)
    a_acc = np.zeros(N)
    
    try:
        for _, row in dataset.iterrows():
            R = np.zeros((agents, N))
            A = np.zeros((agents, N))
            for ai in range(agents):
                agent = PTIBLAgent(*params) if use_pt else IBLAgent(*params)
                seq = agent.run_n(row, N)
                R[ai] = [int(c == 'risky') for c in seq]
                A[ai, 0] = 0
                for t in range(1, N):
                    A[ai, t] = int(seq[t] != seq[t-1])
            r_acc += R.mean(axis=0)
            a_acc += A.mean(axis=0)
    except Exception as e:
        raise RuntimeError(f"Error during simulation: {e}")
    
    return r_acc / P, a_acc / P



