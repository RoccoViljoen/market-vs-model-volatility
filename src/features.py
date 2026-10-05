# The 21-day realised variance target and the baseline forecasts (HV21, EWMA, VIX).

import numpy as np

from src.data import DAYS_PER_YEAR, EWMA_LAMBDA, HORIZON


def log_returns(simple_returns):
    return np.log1p(simple_returns)


def add_forward_realized_variance(df, horizon=HORIZON):
    """Annualised realised variance over days t+1 to t+21 (returns not demeaned)."""
    dates = list(df["date"])
    if dates != sorted(dates):
        raise ValueError("dates not in order")
    if df["log_return"].isna().any():
        raise ValueError("missing log returns")

    df = df.copy()
    squared = df["log_return"] ** 2

    # The sum starts at j = 1, so day t is excluded: the forecast is made at its close.
    total = 0
    for j in range(1, horizon + 1):
        total = total + squared.shift(-j)

    df["target_var"] = (DAYS_PER_YEAR / horizon) * total
    df["target_vol"] = np.sqrt(df["target_var"])
    df["target_end_date"] = df["date"].shift(-horizon)
    return df


def historical_variance(log_return, window):
    return DAYS_PER_YEAR * (log_return ** 2).rolling(window).mean()


def ewma_variance_manual(log_return, lam=EWMA_LAMBDA):
    """EWMA variance for day t+1, made at the close of day t (Hull, 23.2)."""
    if lam <= 0 or lam >= 1:
        raise ValueError("lambda has to be between 0 and 1")
    r2 = (log_return ** 2).to_numpy()

    # A loop is needed because each value depends on the previous one. The
    # starting value matches pandas ewm(adjust=False), which the tests compare against.
    var_next = np.empty(len(r2))
    var_next[0] = r2[0]
    for t in range(1, len(r2)):
        var_next[t] = lam * var_next[t - 1] + (1 - lam) * r2[t]
    return DAYS_PER_YEAR * var_next


def vix_variance(vix):
    # VIX is quoted in volatility points (20 means 20% a year). It is an implied,
    # risk-neutral volatility (Hull, 15.11), so it is expected to sit above
    # realised volatility on average.
    return (vix / 100) ** 2


def build_baseline_forecasts(df, lam=EWMA_LAMBDA):
    if df["log_return"].isna().any():
        raise ValueError("missing log returns")

    df = df.copy()
    df["hv21_var"] = historical_variance(df["log_return"], 21)
    df["ewma_var"] = ewma_variance_manual(df["log_return"], lam)
    df["vix_var"] = vix_variance(df["vix"])
    return df
