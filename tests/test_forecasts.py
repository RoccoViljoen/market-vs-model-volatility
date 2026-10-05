# Tests for the forecasts: hand calculations, a check of the GARCH recursion
# against arch, the QLIKE loss and the train-only regressions.

import numpy as np
import pandas as pd
import pytest

from src.features import ewma_variance_manual, historical_variance, vix_variance
from src.metrics import qlike_var
from src.models import check_params, fit_garch, garch_params
from src.models import garch_horizon_variance, garch_next_variance
from src.models import fit_vol_regression, predict_vol


# Baselines

def test_hv21_by_hand():
    r = pd.Series(np.arange(1, 31) / 1000)
    out = historical_variance(r, 21)
    assert out.iloc[:20].isna().all()
    assert np.isclose(out.iloc[20], 252 * (r.iloc[0:21] ** 2).mean())
    assert np.isclose(out.iloc[21], 252 * (r.iloc[1:22] ** 2).mean())


def test_ewma_three_updates_by_hand():
    # Values calculated by hand with lambda = 0.94.
    r = pd.Series([0.012, -0.008, 0.005, -0.021])
    out = ewma_variance_manual(r)
    assert np.isclose(out[0], 252 * 0.000144)
    assert np.isclose(out[1], 252 * 0.0001392)
    assert np.isclose(out[2], 252 * 0.000132348)
    assert np.isclose(out[3], 252 * 0.00015086712)


def test_ewma_matches_pandas():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 300))
    manual = ewma_variance_manual(r)
    built_in = 252 * (r ** 2).ewm(alpha=0.06, adjust=False).mean()
    assert np.isclose(manual, built_in.to_numpy()).all()


def test_vix_variance_units():
    assert np.isclose(vix_variance(20), 0.04)


# GARCH

def test_garch_next_variance_by_hand():
    p = {"omega": 0.02, "alpha": 0.09, "beta": 0.89}
    # 0.02 + 0.09 * 1.5^2 + 0.89 * 1.2 = 1.2905
    assert np.isclose(garch_next_variance(p, 1.5, 1.2), 1.2905)
    assert np.isclose(garch_next_variance(p, -1.5, 1.2), 1.2905)


def test_percent_to_decimal_units():
    # 1 percent squared per day is 252 / 10000 = 0.0252 per year in decimals.
    p = {"v_long": 1.0, "persistence": 0.98}
    assert np.isclose(garch_horizon_variance(1.0, p), 252 * 1.0 / 10000)


def test_three_step_path_by_hand():
    # With V_L = 1 and a start of 1.5, the path is 1.5, 1 + 0.98 * 0.5, 1 + 0.98^2 * 0.5.
    p = {"v_long": 1.0, "persistence": 0.98}
    out = garch_horizon_variance(1.5, p, horizon=3)
    assert np.isclose(out, 252 * (1.5 + 1.49 + 1.4802) / 3 / 10000)


def test_manual_recursion_matches_arch():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2003-01-02", periods=1500)
    y = pd.Series(rng.standard_t(5, 1500), index=dates)
    res = fit_garch(y)
    p = garch_params(res)
    s2_last = res.conditional_volatility.iloc[-1] ** 2
    s2_next = garch_next_variance(p, y.iloc[-1], s2_last)
    arch_path = res.forecast(horizon=21, reindex=False).variance.iloc[0].to_numpy()

    assert np.isclose(s2_next, arch_path[0])
    arch_annual = 252 * arch_path.mean() / 10000
    assert np.isclose(garch_horizon_variance(s2_next, p), arch_annual)


def test_bad_refit_stops_the_run():
    check_params({"omega": 0.02, "alpha": 0.09, "beta": 0.89, "persistence": 0.98})
    with pytest.raises(ValueError):
        check_params({"omega": 0.02, "alpha": 0.1, "beta": 0.9, "persistence": 1.0})
    with pytest.raises(ValueError):
        check_params({"omega": 0, "alpha": 0.09, "beta": 0.89, "persistence": 0.98})


# QLIKE

def test_perfect_qlike_is_zero():
    target = pd.Series([0.01, 0.04, 0.09])
    assert np.isclose(qlike_var(target, target), 0).all()


def test_qlike_punishes_under_prediction_more():
    target = pd.Series([1.0])
    under = qlike_var(target, pd.Series([0.5])).iloc[0]
    over = qlike_var(target, pd.Series([2.0])).iloc[0]
    assert np.isclose(under, 2 - np.log(2) - 1)
    assert np.isclose(over, 0.5 + np.log(2) - 1)
    assert under > over


def test_qlike_bad_forecast_raises():
    target = pd.Series([0.01, 0.04])
    with pytest.raises(ValueError):
        qlike_var(target, pd.Series([0.01, 0]))
    with pytest.raises(ValueError):
        qlike_var(target, pd.Series([0.01, np.nan]))


# Calibrated VIX and combination

def make_train_frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"vix_vol": rng.uniform(0.12, 0.35, 500),
                         "garch_vol": rng.uniform(0.10, 0.30, 500)})


def test_calibrated_vix_frozen_from_train():
    train = make_train_frame()
    train["target_vol"] = 0.02 + 0.7 * train["vix_vol"]
    results = fit_vol_regression(train, ["vix_vol"])
    assert np.isclose(results.params.to_numpy(), [0.02, 0.7]).all()

    # New rows use the frozen train coefficients; their own targets have no effect.
    new = pd.DataFrame({"vix_vol": [0.15, 0.40], "target_vol": [5.0, 9.0]})
    assert np.isclose(predict_vol(results, new, ["vix_vol"]), [0.125, 0.30]).all()

    with pytest.raises(ValueError):
        predict_vol(results, pd.DataFrame({"vix_vol": [-1.0]}), ["vix_vol"])


def test_combination_recovers_coefficients():
    train = make_train_frame()
    train["target_vol"] = 0.02 + 0.5 * train["vix_vol"] + 0.3 * train["garch_vol"]
    results = fit_vol_regression(train, ["vix_vol", "garch_vol"])
    assert np.isclose(results.params.to_numpy(), [0.02, 0.5, 0.3]).all()
