# Period splits, model evaluation and selection of the disagreement cases.

import pandas as pd

from src.data import CASE_GAP, MODEL_NAMES, N_CASES, PERIODS
from src.metrics import loss_difference_test, mean_error_vol, qlike_var, rmse_vol


def period_masks(df):
    masks = {}
    for name, (start, end) in PERIODS.items():
        date_inside = (df["date"] >= start) & (df["date"] <= end)
        target_inside = df["target_end_date"] <= end
        masks[name] = date_inside & target_inside
    return masks


def common_sample(df, mask, models=MODEL_NAMES):
    # Every model is scored on exactly the same days, so the comparison is fair.
    columns = ["target_var", "target_vol"] + [m + "_var" for m in models]
    return df[mask].dropna(subset=columns)


def evaluate_models(sample, models=MODEL_NAMES):
    target_var = sample["target_var"]
    target_vol = sample["target_vol"]
    rows = []
    for m in models:
        rows.append({"model": m,
                     "n": len(sample),
                     "qlike": qlike_var(target_var, sample[m + "_var"]).mean(),
                     "rmse_vol": rmse_vol(target_vol, sample[m + "_vol"]),
                     "mean_error_vol": mean_error_vol(target_vol, sample[m + "_vol"])})
    return pd.DataFrame(rows)


def primary_test(sample):
    loss_garch = qlike_var(sample["target_var"], sample["garch_var"])
    loss_vix = qlike_var(sample["target_var"], sample["vix_var"])
    return loss_difference_test(loss_garch, loss_vix)


def disagreement_cases(sample, n=N_CASES, gap=CASE_GAP):
    """The n days with the largest VIX vs GARCH disagreement, at least gap days apart."""
    cases = sample.copy()
    cases["vix_minus_garch_vol"] = cases["vix_vol"] - cases["garch_vol"]
    cases["abs_gap"] = cases["vix_minus_garch_vol"].abs()
    cases["qlike_vix"] = qlike_var(cases["target_var"], cases["vix_var"])
    cases["qlike_garch"] = qlike_var(cases["target_var"], cases["garch_var"])
    cases["closer"] = "garch"
    cases.loc[cases["qlike_vix"] < cases["qlike_garch"], "closer"] = "vix"
    cases = cases.sort_values("abs_gap", ascending=False)

    # Row labels count trading days. Without a minimum gap the top cases would
    # be neighbouring days from the same episode.
    chosen = []
    for row in list(cases.index):
        far_enough = True
        for kept in chosen:
            if abs(row - kept) < gap:
                far_enough = False
        if far_enough and len(chosen) < n:
            chosen.append(row)

    cases = cases.loc[chosen].sort_values("date")
    columns = ["date", "target_end_date", "vix", "target_vol", "vix_vol", "garch_vol",
               "vix_minus_garch_vol", "qlike_vix", "qlike_garch", "closer"]
    return cases[columns]
