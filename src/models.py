# Walk-forward GARCH(1,1) forecasts, and the calibrated VIX and VIX + GARCH
# combination, both fitted on the training period only.

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model

from src.data import DAYS_PER_YEAR, FIRST_REFIT, HAC_LAGS, HORIZON, PERIODS
from src.data import REFIT_FREQ


def fit_garch(y):
    """Fit a zero-mean Gaussian GARCH(1,1). arch works best with percent returns."""
    model = arch_model(y, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    res = model.fit(disp="off")
    if res.convergence_flag != 0:
        raise ValueError("GARCH fit didn't converge")
    return res


def garch_params(res):
    omega = res.params["omega"]
    alpha = res.params["alpha[1]"]
    beta = res.params["beta[1]"]
    persistence = alpha + beta
    v_long = np.nan
    if persistence < 1:
        v_long = omega / (1 - persistence)
    return {"omega": omega, "alpha": alpha, "beta": beta,
            "persistence": persistence, "v_long": v_long}


def check_params(p, label=""):
    if p["omega"] <= 0:
        raise ValueError("omega not positive " + label)
    if p["alpha"] < 0 or p["beta"] < 0:
        raise ValueError("negative alpha or beta " + label)
    if p["persistence"] >= 1:
        raise ValueError("alpha + beta >= 1 " + label)


def garch_next_variance(p, r, s2):
    # sigma^2_{t+1} = omega + alpha * r_t^2 + beta * sigma^2_t (Hull, 23.3)
    return p["omega"] + p["alpha"] * r ** 2 + p["beta"] * s2


def garch_horizon_variance(s2_next, p, horizon=HORIZON):
    """Average expected variance over days t+1 to t+21, annualised, in decimals."""
    # E[sigma^2_{t+k}] = V_L + (alpha + beta)^(k-1) * (sigma^2_{t+1} - V_L),
    # following Hull 23.6 but starting from sigma^2_{t+1}.
    k = np.arange(1, horizon + 1)
    path = p["v_long"] + p["persistence"] ** (k - 1) * (s2_next - p["v_long"])
    return DAYS_PER_YEAR * path.mean() / 10000  # percent squared to decimal


def walk_forward_garch(df, start=FIRST_REFIT, end=PERIODS["validation"][1],
                       freq=REFIT_FREQ):
    """Expanding-window GARCH forecasts, refitted at the start of each quarter.

    Each fit uses only returns strictly before the refit date. Parameters are
    then held fixed for the quarter while the variance is updated daily.
    """
    if df["log_return"].isna().any():
        raise ValueError("missing log returns")
    y = 100 * df.set_index("date")["log_return"]
    refit_dates = list(pd.date_range(start, end, freq=freq))

    rows = []
    history = []
    for i in range(len(refit_dates)):
        refit_date = refit_dates[i]
        fit_data = y[y.index < refit_date]
        res = fit_garch(fit_data)
        p = garch_params(res)
        check_params(p, "at refit " + str(refit_date.date()))
        sample_end = fit_data.index.max()

        s2_last = res.conditional_volatility.iloc[-1] ** 2
        s2 = garch_next_variance(p, fit_data.iloc[-1], s2_last)

        history.append({"refit_date": refit_date, "sample_end": sample_end,
                        "nobs": res.nobs, "omega": p["omega"], "alpha": p["alpha"],
                        "beta": p["beta"], "persistence": p["persistence"]})

        in_quarter = (y.index >= refit_date) & (y.index <= end)
        if i + 1 < len(refit_dates):
            in_quarter = in_quarter & (y.index < refit_dates[i + 1])
        quarter = y[in_quarter]

        for date, r in zip(list(quarter.index), quarter.to_numpy()):
            s2_next = garch_next_variance(p, r, s2)
            rows.append({"date": date,
                         "estimation_sample_end": sample_end,
                         "garch_var": garch_horizon_variance(s2_next, p)})
            s2 = s2_next

    forecasts = pd.DataFrame(rows)
    if not np.isfinite(forecasts["garch_var"]).all():
        raise ValueError("missing or infinite GARCH forecasts")
    if (forecasts["garch_var"] <= 0).any():
        raise ValueError("GARCH forecast of zero or below")
    if not (forecasts["estimation_sample_end"] < forecasts["date"]).all():
        raise ValueError("a GARCH fit used data from its own forecast date")
    return forecasts, pd.DataFrame(history)


def fit_vol_regression(train, columns, lags=HAC_LAGS):
    """Fit target_vol = a + b * x once on the training rows.

    columns=["vix_vol"] gives the calibrated VIX, which removes the persistent
    level bias in VIX. columns=["vix_vol", "garch_vol"] gives the combination.
    """
    X_const = sm.add_constant(train[columns])
    model = sm.OLS(train["target_vol"], X_const)
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def predict_vol(results, frame, columns):
    X_const = sm.add_constant(frame[columns], has_constant="add")
    forecast_vol = results.predict(X_const)

    # Volatility must be positive before it is squared for QLIKE.
    if not np.isfinite(forecast_vol).all():
        raise ValueError("missing or infinite vol forecasts")
    if (forecast_vol <= 0).any():
        raise ValueError("vol forecast of zero or below")
    return forecast_vol
