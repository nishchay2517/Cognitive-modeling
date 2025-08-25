import argparse
from pathlib import Path
from iblpt.config import load_config
from iblpt.data.loader import load_dataset
from iblpt.human.metrics import human_r_ts_est
from iblpt.plotting import plot_r_rate_curves


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--model', choices=['ibl','pt'], default='pt')
    p.add_argument('--params', required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    est = load_dataset(cfg.data.estimation)

    txt = Path(args.params).read_text().strip().strip('[]') if Path(args.params).exists() else args.params
    params = tuple(float(x) for x in txt.split(','))

    fig = plot_r_rate_curves(
        est, human_r_ts_est,
        models=[{'label': f'{args.model.upper()} model', 'params': params, 'use_pt': args.model=='pt'}],
        N=cfg.simulation.trials, agents=20
    )
    out = Path(cfg.output.dir) / 'IBLPT_vs_human.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f'Saved figure to {out}')


if __name__ == '__main__':
    main()



