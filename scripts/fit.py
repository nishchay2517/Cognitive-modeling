import argparse
from pathlib import Path
from iblpt.config import load_config
from iblpt.data.loader import load_dataset
from iblpt.optimize import fit_ibl, fit_pt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--model', choices=['ibl','pt'], default='pt')
    args = p.parse_args()

    cfg = load_config(args.config)
    est = load_dataset(cfg.data.estimation)

    if args.model == 'ibl':
        bounds = cfg.opt.ibl_bounds
        res = fit_ibl(bounds, est, N=cfg.simulation.trials, agents=cfg.simulation.agents, r_weight=cfg.metrics.r_weight)
    else:
        bounds = cfg.opt.ibl_bounds + cfg.opt.pt_extra_bounds
        res = fit_pt(bounds, est, N=cfg.simulation.trials, agents=cfg.simulation.agents, r_weight=0.5)

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"fit_{args.model}.txt"
    out.write_text(str(res.x))
    print(f"Saved params to {out}")


if __name__ == '__main__':
    main()


