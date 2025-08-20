import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from repeated.IBL_v2 import IBLAgent
from repeated.IBLPT_v1 import PTIBLAgent
from repeated.human_metrc import *

cols = ['id','val_high','p_high','val_low','val_safe','sure','d1','mode']
est  = pd.read_csv('repeated/data/60estimationset.dat',   sep=r'\s+', header=None, names=cols)
comp = pd.read_csv('repeated/data/60competitionset.dat', sep=r'\s+', header=None, names=cols)
 

def eval_ts(dataset, human_ts, params, use_pt=False, N=100, agents=5):
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

# -----------------------------------------------------------------------------
# 5.  MSD + GA objectives (fit to R-rate series)
# -----------------------------------------------------------------------------
def MSD(x, y):
    return np.mean((x - y)**2)

def fit_ibl(x):
    r_weight=0.9
    r_ts, a_ts = eval_ts(est, human_r_ts_est, x, False, N=100, agents=5)
    return MSD(r_ts, human_r_ts_est)*r_weight + MSD(a_ts,human_a_ts_est)*(1-r_weight)

def fit_pt(x):
    r_weight=0.5
    r_ts, a_ts = eval_ts(est, human_r_ts_est, x, True, N=100, agents=5)
    return MSD(r_ts, human_r_ts_est)*r_weight + MSD(a_ts,human_a_ts_est)*(1-r_weight)

# -----------------------------------------------------------------------------
# 6.  Callbacks to monitor the GA
# -----------------------------------------------------------------------------
ibl_i = 0
def cb_ibl(xk, _):
    global ibl_i
    ibl_i += 1
    print(f"[IBL] Gen {ibl_i:2d}, MSD={fit_ibl(xk):.5f}, params={xk}")
    return False

pt_i = 0
def cb_pt(xk, _):
    global pt_i
    pt_i += 1
    print(f"[PT ] Gen {pt_i:2d}, MSD={fit_pt(xk):.5f}, params={xk}")
    return False

# -----------------------------------------------------------------------------
# 9.  Print summary (MSD / Corr / AIC on R & A, both sets)
# -----------------------------------------------------------------------------
def summarize(label, params, use_pt): 
    print('With params:',params)
    # Estimation
    r_est, a_est = eval_ts(est, human_r_ts_est, params, use_pt, N=100, agents=5)
    # Competition
    r_cmp, a_cmp = eval_ts(comp, human_r_ts_comp, params, use_pt, N=100, agents=5)

    # compute metrics
    def metrics(model_ts, human_ts, k):
        msd = MSD(model_ts, human_ts)
        corr = np.corrcoef(model_ts, human_ts)[0,1]
        aic = 100 * np.log(max(msd,1e-8)) + 2*k
        return msd, corr, aic

    k = 3 + (3 if use_pt else 0)
    msd_re, cr_re, aic_re = metrics(r_est, human_r_ts_est,   k)
    msd_ae, cr_ae, aic_ae = metrics(a_est, human_a_ts_est,   k)
    msd_rc, cr_rc, aic_rc = metrics(r_cmp, human_r_ts_comp,  k)
    msd_ac, cr_ac, aic_ac = metrics(a_cmp, human_a_ts_comp,  k)

    print(f"\n{label}:")
    print(f"  Est R → MSD={msd_re:.4f}, Corr={cr_re:.3f}, AIC={aic_re:.1f}")
    print(f"  Est A → MSD={msd_ae:.4f}, Corr={cr_ae:.3f}, AIC={aic_ae:.1f}")
    print(f"  Cmp R → MSD={msd_rc:.4f}, Corr={cr_rc:.3f}, AIC={aic_rc:.1f}")
    print(f"  Cmp A → MSD={msd_ac:.4f}, Corr={cr_ac:.3f}, AIC={aic_ac:.1f}")

def plot_r_rate_curves(dataset, human_r_ts, models, N=100, agents=20):
    """
    dataset      : the pandas DataFrame of problems (est or comp)
    human_r_ts   : 1D array of length N with human R-rate
    models       : list of dicts: {'label':str, 'params':tuple, 'use_pt':bool}
    N            : number of trials
    agents       : Monte Carlo replications per problem
    """

    trials = np.arange(1, N+1)
    plt.figure(figsize=(10,5))
    # plot human
    plt.plot(trials, human_r_ts, 'k-', lw=2, label="Human R-rate")
    # plot each model
    for m in models:
        r_ts, _ = eval_ts(dataset, human_r_ts, m['params'], m['use_pt'], N=N, agents=agents)
        plt.plot(trials, r_ts, lw=1.5, alpha=0.8, label=m['label'])
    plt.xlabel("Trial")
    plt.ylabel("R-rate")
    plt.title("R-rate over Trials")
    plt.legend()
    plt.tight_layout()
    plt.savefig("IBLPT_vs_human.png")
    plt.show()

# -----------------------------------------------------------------------------
# 7.  Fit on estimation set (much faster now)
# -----------------------------------------------------------------------------
if __name__=="__main__":
    bnds_ibl = [(0.01, 10), (0.01, 10), (0.0, 1.0)]
    # print('Fitting for IBL')
    # res_ibl = differential_evolution(
    #     fit_ibl, bnds_ibl,
    #     maxiter=15, popsize=8,
    #     updating='deferred', workers=-1,
    #     tol=1e-2, callback=cb_ibl
    # )

    res_ibl={'x':[4.59856917,0.04554824,0.01635943]}

    print('Fitting for PT+IBL')
    bnds_pt = bnds_ibl + [(0.2, 1.0), (0.2, 1.0), (0.1, 5.0)]
    res_pt = differential_evolution(
        fit_pt, bnds_pt,
        maxiter=15, popsize=8,
        updating='deferred', workers=-1,
        tol=1e-2, callback=cb_pt
    )

    # -----------------------------------------------------------------------------
    # 8.  Default + fitted parameters
    # -----------------------------------------------------------------------------
    default = (5.27, 1.46, 0.09)
    # ibl_opt = tuple(res_ibl.x)
    pt_opt  = tuple(res_pt.x)


    print("\n=== FINAL RESULTS ===")
    summarize("Default IBL",     default, False)
    # summarize("Optimized IBL",    ibl_opt, False)
    summarize("PT-Optimized IBL", pt_opt,   True)
    plot_r_rate_curves(
    est, human_r_ts_est,
        models=[
        {'label':'PT-Optimized IBL', 'params':pt_opt,  'use_pt':True}
        ],
        N=100,
        agents=20
    )