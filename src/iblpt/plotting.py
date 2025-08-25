import numpy as np
import matplotlib.pyplot as plt
from iblpt.simulate import eval_ts


def plot_r_rate_curves(dataset, human_r_ts, models, N=100, agents=20, save_path="IBLPT_vs_human.png"):
    """
    Plot R-rate curves for human data vs model predictions.
    
    Args:
        dataset: pandas DataFrame of problems (est or comp)
        human_r_ts: 1D array of length N with human R-rate
        models: list of dicts: {'label':str, 'params':tuple, 'use_pt':bool}
        N: number of trials
        agents: Monte Carlo replications per problem
        save_path: path to save the figure
    """
    trials = np.arange(1, N+1)
    fig, ax = plt.subplots(figsize=(10,5))
    
    # plot human
    ax.plot(trials, human_r_ts, 'k-', lw=2, label="Human R-rate")
    
    # plot each model
    for m in models:
        r_ts, _ = eval_ts(dataset, m['params'], m['use_pt'], N=N, agents=agents)
        ax.plot(trials, r_ts, lw=1.5, alpha=0.8, label=m['label'])
    
    ax.set_xlabel("Trial")
    ax.set_ylabel("R-rate")
    ax.set_title("R-rate over Trials")
    ax.legend()
    fig.tight_layout()
    
    # Save the figure
    plt.savefig(save_path)
    plt.show()
    
    return fig



