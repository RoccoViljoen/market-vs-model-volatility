# Runs the full study from the raw files:  python -m scripts.run_pipeline
#
# With OPEN_HOLDOUT = False (in src/data.py), forecasts stop at the end of 2019
# and only validation is scored. Setting it to True reruns the published test period.

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import CRSP_PATH, VIX_PATH, RESULTS_FOLDER, MODEL_NAMES, OPEN_HOLDOUT, PERIODS
from src.data import clean_market_data, load_crsp_csv, load_vix_csv
from src.features import add_forward_realized_variance, build_baseline_forecasts, log_returns
from src.metrics import calibration_regression, encompassing_regression
from src.models import fit_vol_regression, predict_vol, walk_forward_garch
from src.validation import common_sample, disagreement_cases, evaluate_models
from src.validation import period_masks, primary_test

SHORT_NAMES = {"hv21": "HV21", "ewma": "EWMA", "garch": "GARCH", "vix": "VIX",
               "vix_cal": "VIX cal.", "combination": "Comb."}


def prepare(crsp, vix):
    df = clean_market_data(crsp, vix)
    df["log_return"] = log_returns(df["sprtrn"])
    df = add_forward_realized_variance(df)
    return build_baseline_forecasts(df)


def add_garch(df, forecast_end):
    forecasts, history = walk_forward_garch(df, end=forecast_end)
    df = pd.merge(df, forecasts[["date", "garch_var"]], on="date", how="left",
                  validate="one_to_one")

    print("GARCH refits:", len(history),
          "| persistence from", round(history["persistence"].min(), 3),
          "to", round(history["persistence"].max(), 3))
    return df


def add_volatilities(df):
    df = df.copy()
    for name in ["hv21", "ewma", "garch", "vix"]:
        df[name + "_vol"] = np.sqrt(df[name + "_var"])
    return df


def add_train_fits(df, masks):
    """Fit the calibrated VIX and the combination once on train, then freeze them."""
    train = df[masks["train"]].dropna(subset=["target_vol", "vix_vol", "garch_vol"])
    calibration = fit_vol_regression(train, ["vix_vol"])
    combination = fit_vol_regression(train, ["vix_vol", "garch_vol"])

    df = df.copy()
    has_vix = df["vix_vol"].notna()
    has_both = has_vix & df["garch_vol"].notna()
    df.loc[has_vix, "vix_cal_vol"] = predict_vol(calibration, df[has_vix], ["vix_vol"])
    df.loc[has_both, "combination_vol"] = predict_vol(combination, df[has_both],
                                                      ["vix_vol", "garch_vol"])
    df["vix_cal_var"] = df["vix_cal_vol"] ** 2
    df["combination_var"] = df["combination_vol"] ** 2

    rows = [{"period": "train", "item": "fit_rows", "value": len(train)}]
    for name, results in [("vix_cal", calibration), ("combination", combination)]:
        for term in list(results.params.index):
            rows.append({"period": "train", "item": name + "_" + term,
                         "value": results.params[term]})
            rows.append({"period": "train", "item": name + "_" + term + "_hac_se",
                         "value": results.bse[term]})
    print("Train fit rows:", len(train), "| last train target ends:",
          train["target_end_date"].max().date())
    return df, rows


def summary_rows(period, prefix, results):
    rows = []
    for key, value in results.items():
        rows.append({"period": period, "item": prefix + key, "value": value})
    return rows


def score_period(df, masks, period):
    sample = common_sample(df, masks[period])
    dropped = masks[period].sum() - len(sample)
    print(period, "| rows scored:", len(sample), "| dropped (missing forecast):", dropped)

    metrics = evaluate_models(sample)
    rows = [{"period": period, "item": "full_label_rows", "value": masks[period].sum()},
            {"period": period, "item": "scored_rows", "value": len(sample)}]
    for i in range(len(metrics)):
        m = metrics["model"].iloc[i]
        for column in ["qlike", "rmse_vol", "mean_error_vol"]:
            rows.append({"period": period, "item": column + "_" + m,
                         "value": metrics[column].iloc[i]})

    for m in MODEL_NAMES:
        c = calibration_regression(sample["target_vol"], sample[m + "_vol"])
        rows = rows + summary_rows(period, "calibration_" + m + "_", c)

    rows = rows + summary_rows(period, "garch_minus_vix_", primary_test(sample))
    rows = rows + summary_rows(period, "encompassing_", encompassing_regression(sample))
    return sample, metrics, rows


def check_validation_unchanged(summary, folder):
    # Extending the forecasts to 2025 must not change any pre-2020 result.
    path = folder + "/summary.csv"
    if not os.path.exists(path):
        return
    saved = pd.read_csv(path)
    saved = saved[saved["period"].isin(["train", "validation"])]
    now = pd.DataFrame(summary)
    now = now[now["period"].isin(["train", "validation"])]
    if list(saved["item"]) != list(now["item"]):
        raise ValueError("train/validation rows in the summary have changed")
    if not np.isclose(saved["value"].to_numpy(), now["value"].to_numpy()).all():
        raise ValueError("train/validation numbers changed after opening the test period")
    print("Train and validation results unchanged")


def make_figures(test_sample, metrics_by_period, folder):
    figures = folder + "/figures"
    os.makedirs(figures, exist_ok=True)
    dates = test_sample["date"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, test_sample["target_vol"], label="Realised, next 21 days")
    ax.plot(dates, test_sample["vix_vol"], label="VIX/100")
    ax.plot(dates, test_sample["garch_vol"], label="Gaussian GARCH")
    ax.plot(dates, test_sample["combination_vol"], label="Combination")
    ax.set_title("Test period 2020-2025: forecasts and realised volatility")
    ax.set_xlabel("Forecast date")
    ax.set_ylabel("Annualised volatility")
    ax.legend()
    fig.savefig(figures + "/holdout_volatility.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    titles = {"validation": "Validation 2016-2019", "test": "Test 2020-2025"}
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, period in zip(axes, ["validation", "test"]):
        metrics = metrics_by_period[period]
        ax.bar(list(metrics["model"].map(SHORT_NAMES)), metrics["qlike"])
        ax.set_title(titles[period])
        ax.set_ylabel("Mean QLIKE")
    axes[1].set_xlabel("Model (lower is better)")
    fig.savefig(figures + "/qlike_validation_vs_test.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(crsp, vix, holdout, folder):
    os.makedirs(folder, exist_ok=True)

    forecast_end = PERIODS["validation"][1]
    scored = ["validation"]
    if holdout:
        forecast_end = PERIODS["test"][1]
        scored = ["validation", "test"]

    df = prepare(crsp, vix)
    df = add_garch(df, forecast_end)
    df = add_volatilities(df)
    masks = period_masks(df)
    df, summary = add_train_fits(df, masks)

    samples = {}
    metrics_by_period = {}
    for period in scored:
        sample, metrics, rows = score_period(df, masks, period)
        samples[period] = sample
        metrics_by_period[period] = metrics
        summary = summary + rows

    if holdout:
        check_validation_unchanged(summary, folder)

    pd.DataFrame(summary).to_csv(folder + "/summary.csv", index=False)

    if holdout:
        cases = disagreement_cases(samples["test"])
        cases.to_csv(folder + "/disagreement_cases.csv", index=False)
        make_figures(samples["test"], metrics_by_period, folder)
    print("Forecasts end", forecast_end, "| saved to", folder)


def main():
    # The published version is rerunnable after the holdout has been opened.
    crsp = load_crsp_csv(CRSP_PATH)
    vix = load_vix_csv(VIX_PATH)
    run(crsp, vix, OPEN_HOLDOUT, RESULTS_FOLDER)


if __name__ == "__main__":
    main()
