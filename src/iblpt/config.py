from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Paths:
    estimation: str
    competition: str


@dataclass
class Simulation:
    trials: int = 100
    agents: int = 5
    seed: int | None = None


@dataclass
class MetricsCfg:
    r_weight: float = 0.9


@dataclass
class OptCfg:
    ibl_bounds: list[list[float]] = None
    pt_extra_bounds: list[list[float]] = None
    maxiter: int = 15
    popsize: int = 8
    tol: float = 1e-2
    workers: int = -1


@dataclass
class OutputCfg:
    dir: str = "outputs"


@dataclass
class Config:
    data: Paths
    simulation: Simulation
    metrics: MetricsCfg
    opt: OptCfg
    output: OutputCfg


def load_config(path: str | Path) -> Config:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    return Config(
        data=Paths(**raw['data']),
        simulation=Simulation(**raw.get('simulation', {})),
        metrics=MetricsCfg(**raw.get('metrics', {})),
        opt=OptCfg(**raw.get('opt', {})),
        output=OutputCfg(**raw.get('output', {})),
    )


