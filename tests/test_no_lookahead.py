# No-lookahead tests: corrupting all returns and VIX values after a cutoff
# must not change any forecast made on or before it.

import numpy as np
import pandas as pd

from src.features import build_baseline_forecasts
from src.models import walk_forward_garch

CUTOFF = "2005-08-15"


def make_fake_frame():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2003-01-02", "2005-12-30")
    df = pd.DataFrame({"date": dates,
                       "log_return": rng.standard_t(5, len(dates)) * 0.01,
                       "vix": rng.uniform(12, 35, len(dates))})
    return df


def corrupt_after(df, cutoff):
    corrupt = df.copy()
    after_T = corrupt["date"] > cutoff
    corrupt.loc[after_T, "log_return"] = 0.5
    corrupt.loc[after_T, "vix"] = 500.0
    return corrupt


def test_baselines_unchanged_through_cutoff():
    df = make_fake_frame()
    clean = build_baseline_forecasts(df)
    corrupt = build_baseline_forecasts(corrupt_after(df, CUTOFF))

    # Starts in April 2003 so that the HV21 window is full.
    through_T = (df["date"] >= "2003-04-01") & (df["date"] <= CUTOFF)
    after_T = df["date"] > CUTOFF
    for col in ["hv21_var", "ewma_var", "vix_var"]:
        assert (clean.loc[through_T, col] == corrupt.loc[through_T, col]).all()
        assert (clean.loc[after_T, col] != corrupt.loc[after_T, col]).all()


def test_garch_unchanged_through_cutoff():
    df = make_fake_frame()
    clean_fc, _ = walk_forward_garch(df, end="2005-09-30")
    corrupt_fc, _ = walk_forward_garch(corrupt_after(df, CUTOFF), end="2005-09-30")

    through_T = clean_fc["date"] <= CUTOFF
    after_T = clean_fc["date"] > CUTOFF
    clean_var = clean_fc["garch_var"]
    corrupt_var = corrupt_fc["garch_var"]
    assert (clean_var[through_T] == corrupt_var[through_T]).all()
    assert (clean_var[after_T] != corrupt_var[after_T]).all()
