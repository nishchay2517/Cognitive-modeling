# MTP: IBL / PT-IBL Refactor

A streamlined package for simulating, fitting, evaluating, and plotting IBL and PT-IBL models.

## Structure
```
src/iblpt/
  agents/           # IBL and PT-IBL agents
  data/             # dataset loader
  human/            # human reference time-series arrays
  simulate.py       # eval_ts
  metrics.py        # msd, corr, aic
  optimize.py       # fit_ibl, fit_pt
  plotting.py       # plot_r_rate_curves
  config.py         # YAML config loader
configs/
  default.yaml      # data paths, trials, agents, bounds, output dir
scripts/
  fit.py            # fit parameters via DE
  evaluate.py       # compute MSD/Corr/AIC
  plot_r_rate.py    # plot R-rate curves
outputs/            # results and figures
repeated/data/      # existing datasets (can be moved later)
```

## Installation
```bash
pip install -e .
```

## Configuration
Edit `configs/default.yaml` to set dataset paths, trials, agents, optimization bounds, and output directory.

## Commands
- Fit parameters:
```bash
python scripts/fit.py --config configs/default.yaml --model pt
python scripts/fit.py --config configs/default.yaml --model ibl
```
- Evaluate metrics (using saved or inline params):
```bash
python scripts/evaluate.py --config configs/default.yaml --model pt --params outputs/fit_pt.txt
python scripts/evaluate.py --config configs/default.yaml --model ibl --params 5.27,1.46,0.09
```
- Plot R-rate curve:
```bash
python scripts/plot_r_rate.py --config configs/default.yaml --model pt --params outputs/fit_pt.txt
```

## Notebook
Use `notebooks/` to keep notebooks that call into the `iblpt` APIs. The existing `eval_with_plot.ipynb` can be updated to import from `iblpt`.

## Next
- Optionally relocate `repeated/data/*` to `data/` and update `configs/default.yaml`.
- Add tests under `tests/`.
