# Runs the full pipeline on simulated data, so it can be run without CRSP access:
#   python -m scripts.run_demo
# Outputs are saved to results/synthetic_demo and are not empirical results.

import os

import numpy as np
import pandas as pd

from scripts.run_pipeline import run
from src.data import DEMO_FOLDER


def make_synthetic_data(seed=0):
    """Simulated CRSP and VIX data in the same format as the loaders return.

    Returns come from a GARCH(1,1) with t(6) shocks. VIX is the true vol
    times 1.15 (roughly like a variance risk premium) plus some noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2003-01-02", "2025-12-31")
    n = len(dates)

    s2 = np.empty(n)
    r = np.empty(n)
    s2[0] = 0.0001
    for t in range(n):
        if t > 0:
            s2[t] = 0.000002 + 0.08 * r[t - 1] ** 2 + 0.9 * s2[t - 1]
        # Scaled so the t(6) shocks have unit variance (a t(6) has variance 1.5).
        r[t] = np.sqrt(s2[t]) * rng.standard_t(6) / np.sqrt(1.5)

    vix_close = 100 * np.sqrt(252 * s2) * 1.15 * (1 + rng.normal(0, 0.05, n))
    crsp = pd.DataFrame({"date": dates, "sprtrn": r})
    vix = pd.DataFrame({"date": dates, "vix": vix_close})

    # One missing VIX value, as occurs in the real data.
    vix = vix[vix["date"] != "2022-06-15"]
    return crsp, vix


def run_demo(folder=DEMO_FOLDER):
    os.makedirs(folder, exist_ok=True)
    crsp, vix = make_synthetic_data()
    run(crsp, vix, False, folder)
    run(crsp, vix, True, folder)


if __name__ == "__main__":
    run_demo()
