import numpy as np
import matplotlib.pyplot as plt
from iblpt.simulate import eval_ts


def plot_r_rate_curves(dataset, human_r_ts, models, N=100, agents=20):
    trials = np.arange(1, N+1)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(trials, human_r_ts, 'k-', lw=2, label="Human R-rate")
    for m in models:
        r_ts, _ = eval_ts(dataset, m['params'], m['use_pt'], N=N, agents=agents)
        ax.plot(trials, r_ts, lw=1.5, alpha=0.8, label=m['label'])
    ax.set_xlabel("Trial")
    ax.set_ylabel("R-rate")
    ax.set_title("R-rate over Trials")
    ax.legend()
    fig.tight_layout()
    return fig


