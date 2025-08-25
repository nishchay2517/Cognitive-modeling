import numpy as np
from iblpt.simulate import eval_ts
from iblpt.human.metrics import human_r_ts_est, human_a_ts_est, human_r_ts_comp, human_a_ts_comp


def msd(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((x - y) ** 2))


def corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def aic_from_msd(msd_value: float, k: int) -> float:
    return float(100 * np.log(max(msd_value, 1e-8)) + 2 * k)


def compute_metrics(model_ts, human_ts, k):
    """Compute MSD, correlation, and AIC for model vs human time series."""
    msd_val = msd(model_ts, human_ts)
    corr_val = corr(model_ts, human_ts)
    aic_val = aic_from_msd(msd_val, k)
    return msd_val, corr_val, aic_val


def summarize(label, params, use_pt, est_dataset, comp_dataset): 
    """Print summary of model performance on estimation and competition sets."""
    print('With params:', params)
    
    # Estimation
    r_est, a_est = eval_ts(est_dataset, params, use_pt, N=100, agents=5)
    # Competition
    r_cmp, a_cmp = eval_ts(comp_dataset, params, use_pt, N=100, agents=5)

    k = 3 + (3 if use_pt else 0)
    msd_re, cr_re, aic_re = compute_metrics(r_est, human_r_ts_est, k)
    msd_ae, cr_ae, aic_ae = compute_metrics(a_est, human_a_ts_est, k)
    msd_rc, cr_rc, aic_rc = compute_metrics(r_cmp, human_r_ts_comp, k)
    msd_ac, cr_ac, aic_ac = compute_metrics(a_cmp, human_a_ts_comp, k)

    print(f"\n{label}:")
    print(f"  Est R → MSD={msd_re:.4f}, Corr={cr_re:.3f}, AIC={aic_re:.1f}")
    print(f"  Est A → MSD={msd_ae:.4f}, Corr={cr_ae:.3f}, AIC={aic_ae:.1f}")
    print(f"  Cmp R → MSD={msd_rc:.4f}, Corr={cr_rc:.3f}, AIC={aic_rc:.1f}")
    print(f"  Cmp A → MSD={msd_ac:.4f}, Corr={cr_ac:.3f}, AIC={aic_ac:.1f}")



