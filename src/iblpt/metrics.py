import numpy as np


def msd(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((x - y) ** 2))


def corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def aic_from_msd(msd_value: float, k: int) -> float:
    return float(100 * np.log(max(msd_value, 1e-8)) + 2 * k)


