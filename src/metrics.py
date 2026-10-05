# Loss functions, calibration regressions and the two HAC-based tests.

import numpy as np
import statsmodels.api as sm

from src.data import HAC_LAGS


def check_forecasts(forecast_var):
    if not np.isfinite(forecast_var).all():
        raise ValueError("forecast has missing or infinite values")
    if (forecast_var <= 0).any():
        raise ValueError("forecast has values of zero or below")


def qlike_var(target_var, forecast_var):
    """QLIKE loss on variance: RV/h - log(RV/h) - 1.

    It is zero for a perfect forecast and penalises under-prediction more
    than over-prediction.
    """
    check_forecasts(forecast_var)
    ratio = target_var / forecast_var
    return ratio - np.log(ratio) - 1


def rmse_vol(target_vol, forecast_vol):
    return np.sqrt(((forecast_vol - target_vol) ** 2).mean())


def mean_error_vol(target_vol, forecast_vol):
    # A positive value means the forecast is too high on average.
    return (forecast_vol - target_vol).mean()


def hac_ols(y, X, lags=HAC_LAGS):
    X_const = sm.add_constant(X)
    return sm.OLS(y, X_const).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def calibration_regression(target_vol, forecast_vol, lags=HAC_LAGS):
    # Descriptive only. A perfectly calibrated forecast gives a = 0 and b = 1.
    res = hac_ols(target_vol, forecast_vol, lags)
    return {"a": res.params.iloc[0], "b": res.params.iloc[1]}


def loss_difference_test(loss_model, loss_benchmark, lags=HAC_LAGS):
    """Mean daily loss difference (model minus benchmark) with a HAC t-statistic.

    Regressing the differences on a constant gives their mean, and the HAC
    standard error allows for the overlap between targets. A positive mean
    difference means the benchmark had the lower loss.
    """
    d = (loss_model - loss_benchmark).to_numpy()
    res = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"mean_diff": res.params[0], "t": res.tvalues[0],
            "p": res.pvalues[0], "n": len(d)}


def encompassing_regression(sample, lags=HAC_LAGS):
    """Regress realised variance on VIX and GARCH variance together.

    If b_vix remains significant with GARCH included, VIX contains
    information that GARCH does not (and similarly for b_garch).
    """
    res = hac_ols(sample["target_var"], sample[["vix_var", "garch_var"]], lags)
    return {"a": res.params.iloc[0],
            "b_vix": res.params.iloc[1],
            "b_garch": res.params.iloc[2],
            "se_vix": res.bse.iloc[1],
            "se_garch": res.bse.iloc[2],
            "t_vix": res.tvalues.iloc[1],
            "t_garch": res.tvalues.iloc[2],
            "p_vix": res.pvalues.iloc[1],
            "p_garch": res.pvalues.iloc[2],
            "r2": res.rsquared,
            "n": len(sample)}
