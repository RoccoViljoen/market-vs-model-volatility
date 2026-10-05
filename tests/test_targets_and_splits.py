# Tests for the target timing, the period splits and a full run on simulated data.

import numpy as np
import pandas as pd

from scripts.run_demo import run_demo
from src.data import DEMO_FOLDER, MODEL_NAMES, N_CASES, PERIODS
from src.features import add_forward_realized_variance
from src.validation import period_masks


# Target timing

def make_spike_frame():
    # 60 days of zero returns with a single 10% return on row 30.
    dates = pd.bdate_range("2024-01-01", periods=60)
    df = pd.DataFrame({"date": dates, "log_return": np.zeros(60)})
    df.loc[30, "log_return"] = 0.1
    return df


def test_day_t_excluded():
    out = add_forward_realized_variance(make_spike_frame())
    assert out.loc[30, "target_var"] == 0


def test_day_t_plus_21_included():
    out = add_forward_realized_variance(make_spike_frame())
    # For row 9 the spike is day t+21, and for row 29 it is day t+1.
    assert np.isclose(out.loc[9, "target_var"], (252 / 21) * 0.1 ** 2)
    assert np.isclose(out.loc[29, "target_var"], (252 / 21) * 0.1 ** 2)


def test_day_t_plus_22_excluded():
    out = add_forward_realized_variance(make_spike_frame())
    assert out.loc[8, "target_var"] == 0


def test_last_21_targets_missing():
    out = add_forward_realized_variance(make_spike_frame())
    assert out["target_var"].iloc[-21:].isna().all()
    assert out["target_var"].iloc[:-21].notna().all()


def test_target_end_date():
    df = make_spike_frame()
    out = add_forward_realized_variance(df)
    assert out.loc[0, "target_end_date"] == df.loc[21, "date"]
    assert out["target_end_date"].iloc[-21:].isna().all()


# Period splits

def make_fake_frame():
    dates = pd.bdate_range("2003-01-02", "2025-12-31")
    df = pd.DataFrame({"date": dates, "log_return": np.ones(len(dates)) * 0.01})
    return add_forward_realized_variance(df)


def test_full_labels_inside_each_period():
    df = make_fake_frame()
    masks = period_masks(df)
    for name, (start, end) in PERIODS.items():
        rows = df[masks[name]]
        assert (rows["date"] >= start).all()
        assert (rows["target_end_date"] <= end).all()


def test_late_2019_labels_excluded():
    df = make_fake_frame()
    masks = period_masks(df)
    enters_2020 = (df["date"] <= "2019-12-31") & (df["target_end_date"] > "2019-12-31")
    assert enters_2020.sum() == 21
    assert not (masks["validation"] & enters_2020).any()
    assert not (masks["test"] & enters_2020).any()


def test_no_train_label_enters_validation():
    df = make_fake_frame()
    masks = period_masks(df)
    assert (df.loc[masks["train"], "target_end_date"] < "2016-01-01").all()


def test_periods_do_not_overlap():
    df = make_fake_frame()
    masks = period_masks(df)
    assert not (masks["train"] & masks["validation"]).any()
    assert not (masks["validation"] & masks["test"]).any()
    assert not (masks["train"] & masks["test"]).any()


def test_chronology():
    df = make_fake_frame()
    masks = period_masks(df)
    train_dates = df.loc[masks["train"], "date"]
    validation_dates = df.loc[masks["validation"], "date"]
    test_dates = df.loc[masks["test"], "date"]
    assert train_dates.max() < validation_dates.min()
    assert validation_dates.max() < test_dates.min()


# Full run on simulated data (takes about a minute)

def test_demo_end_to_end():
    run_demo(DEMO_FOLDER)

    summary = pd.read_csv(DEMO_FOLDER + "/summary.csv")
    for period in ["validation", "test"]:
        rows = summary[summary["period"] == period].set_index("item")["value"]
        assert rows["scored_rows"] > 0
        assert rows["scored_rows"] <= rows["full_label_rows"]
        for m in MODEL_NAMES:
            assert np.isfinite(rows["qlike_" + m])
        assert "garch_minus_vix_mean_diff" in rows.index
        assert "encompassing_b_vix" in rows.index

    train_items = list(summary[summary["period"] == "train"]["item"])
    assert "vix_cal_vix_vol" in train_items
    assert "combination_garch_vol" in train_items

    cases = pd.read_csv(DEMO_FOLDER + "/disagreement_cases.csv")
    assert len(cases) == N_CASES
    assert (pd.to_datetime(cases["date"]) >= PERIODS["test"][0]).all()
