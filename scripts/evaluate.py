import argparse
from pathlib import Path
import numpy as np
from iblpt.config import load_config
from iblpt.data.loader import load_dataset
from iblpt.simulate import eval_ts
from iblpt.metrics import msd, corr, aic_from_msd
from iblpt.human.metrics import human_r_ts_est, human_a_ts_est, human_r_ts_comp, human_a_ts_comp


def metrics_block(model_ts, human_ts, k):
    m = msd(model_ts, human_ts)
    c = corr(model_ts, human_ts)
    a = aic_from_msd(m, k)
    return m, c, a


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--model', choices=['ibl','pt'], default='pt')
    p.add_argument('--params', required=True, help='comma-separated params or path to file')
    args = p.parse_args()

    cfg = load_config(args.config)
    est = load_dataset(cfg.data.estimation)
    comp = load_dataset(cfg.data.competition)

    if Path(args.params).exists():
        txt = Path(args.params).read_text().strip().strip('[]')
    else:
        txt = args.params
    params = tuple(float(x) for x in txt.split(','))

    use_pt = args.model == 'pt'
    k = 3 + (3 if use_pt else 0)

    r_est, a_est = eval_ts(est, params, use_pt, N=cfg.simulation.trials, agents=cfg.simulation.agents)
    r_cmp, a_cmp = eval_ts(comp, params, use_pt, N=cfg.simulation.trials, agents=cfg.simulation.agents)

    msd_re, cr_re, aic_re = metrics_block(r_est, human_r_ts_est,   k)
    msd_ae, cr_ae, aic_ae = metrics_block(a_est, human_a_ts_est,   k)
    msd_rc, cr_rc, aic_rc = metrics_block(r_cmp, human_r_ts_comp,  k)
    msd_ac, cr_ac, aic_ac = metrics_block(a_cmp, human_a_ts_comp,  k)

    print(f"Est R → MSD={msd_re:.4f}, Corr={cr_re:.3f}, AIC={aic_re:.1f}")
    print(f"Est A → MSD={msd_ae:.4f}, Corr={cr_ae:.3f}, AIC={aic_ae:.1f}")
    print(f"Cmp R → MSD={msd_rc:.4f}, Corr={cr_rc:.3f}, AIC={aic_rc:.1f}")
    print(f"Cmp A → MSD={msd_ac:.4f}, Corr={cr_ac:.3f}, AIC={aic_ac:.1f}")


if __name__ == '__main__':
    main()


