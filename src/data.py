# Project settings, and loading and cleaning of the CRSP and VIX data.
# All fixed choices are defined here so they can be found in one place.

import numpy as np
import pandas as pd

CRSP_PATH = "data/raw/crsp_sp500_daily.csv"
VIX_PATH = "data/raw/vix_history.csv"
RESULTS_FOLDER = "results"
DEMO_FOLDER = "results/synthetic_demo"

DATA_START = "2003-01-02"
DATA_END = "2025-12-31"

# A forecast date belongs to a period only if its full 21-day target window
# also ends inside that period, so no target crosses a boundary.
PERIODS = {
    "train": ("2004-01-01", "2015-12-31"),
    "validation": ("2016-01-01", "2019-12-31"),
    "test": ("2020-01-01", "2025-12-31"),
}

HORIZON = 21
DAYS_PER_YEAR = 252
EWMA_LAMBDA = 0.94
FIRST_REFIT = "2004-01-01"  # the first GARCH fit uses 2003 (warm-up) only
REFIT_FREQ = "QS"
HAC_LAGS = 21  # adjacent 21-day targets share 20 days, so errors are autocorrelated

MODEL_NAMES = ["hv21", "ewma", "garch", "vix", "vix_cal", "combination"]

N_CASES = 3
CASE_GAP = 63

# The test period was opened once, after the validation results were saved.
# It stays True now so the published results can be rerun. Set it to False to
# run validation only.
OPEN_HOLDOUT = True


def load_crsp_csv(path):
    """Load the date and S&P 500 price-only return from the WRDS file."""
    df = pd.read_csv(path)

    # Column names differ between the new (CIZ) and old CRSP formats.
    names = {"DlyCalDt": "date", "dlycaldt": "date", "caldt": "date", "CALDT": "date",
             "DlyPrcRet": "sprtrn", "dlyprcret": "sprtrn", "SPRTRN": "sprtrn"}
    df = df.rename(columns=names)
    if "date" not in df.columns or "sprtrn" not in df.columns:
        raise ValueError("can't find the date/return columns: " + str(list(df.columns)))
    df = df[["date", "sprtrn"]]

    # Assumes YYYY-MM-DD dates, which can be selected when downloading from WRDS.
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_vix_csv(path):
    df = pd.read_csv(path, usecols=["DATE", "CLOSE"])
    df = df.rename(columns={"DATE": "date", "CLOSE": "vix"})
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    return df


def validate_unique_dates(dates):
    dates = list(dates)
    if len(dates) != len(set(dates)):
        raise ValueError("duplicate dates")
    if dates != sorted(dates):
        raise ValueError("dates not in order")


def clean_market_data(crsp, vix, start=DATA_START, end=DATA_END):
    """Merge VIX onto the CRSP trading days and check the data for errors."""
    for name, frame in [("CRSP", crsp), ("VIX", vix)]:
        if frame.duplicated(subset=["date"]).sum() > frame.duplicated().sum():
            raise ValueError(name + " has two different rows for the same date")

    crsp = crsp.drop_duplicates(subset=["date"]).sort_values("date")
    vix = vix.drop_duplicates(subset=["date"]).sort_values("date")

    if not np.isfinite(crsp["sprtrn"]).all():
        raise ValueError("missing returns in CRSP")
    if (crsp["sprtrn"] <= -1).any():
        raise ValueError("return of -100% or worse in CRSP")

    # Left join on the CRSP calendar: a missing VIX value must not remove a
    # return from anyone's 21-day target window.
    df = pd.merge(crsp, vix, on="date", how="left", validate="one_to_one")
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df = df.reset_index()
    del df["index"]

    vix_values = df["vix"].dropna()
    if not np.isfinite(vix_values).all():
        raise ValueError("infinite VIX values")
    if (vix_values <= 0).any():
        raise ValueError("VIX of zero or below")

    validate_unique_dates(df["date"])
    print("Rows:", len(df), "| Missing VIX:", df["vix"].isna().sum())
    return df
