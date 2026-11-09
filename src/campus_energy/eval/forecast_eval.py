"""Forecast model evaluation harness.

Evaluates load forecasting models on a strict temporal holdout:
- First 21 days: Training context
- Last 7 days: Held-out evaluation (zero future leakage)

Models evaluated:
1. Ridge regression (ExpectedConsumptionModel baseline)
2. Meta Prophet (generalized additive time-series model)
3. Seasonal-Naive (t - 7 days 'same slot last week' baseline)

Metrics:
- MAE (Mean Absolute Error, in kW)
- MAPE (Mean Absolute Percentage Error, in %)
- RMSE (Root Mean Squared Error, in kW)
Segmented by day type (overall, weekday, weekend, holiday).
"""
from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Workaround for polars CPU flag compatibility with cmdstanpy Stan CSV parsing
try:
    import polars
except Exception:
    import builtins
    _real_import = builtins.__import__
    def _polars_guard(name, *args, **kwargs):
        if name == "polars":
            raise ImportError("polars disabled due to CPU compatibility")
        return _real_import(name, *args, **kwargs)
    builtins.__import__ = _polars_guard

from prophet import Prophet

from campus_energy.analytics.baseline import ExpectedConsumptionModel
from campus_energy.eval.baselines import SeasonalNaiveForecaster

logger = logging.getLogger(__name__)


def prepare_forecast_data(
    data_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, set[pd.Timestamp]]:
    """Load campus meter telemetry, context, and identify holiday calendar."""
    if data_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        data_dir = base_dir / "data" / "generated"
    else:
        data_dir = Path(data_dir)

    tel_path = data_dir / "telemetry.csv"
    ctx_path = data_dir / "context.csv"
    assets_path = data_dir / "assets.csv"

    if not all(p.exists() for p in (tel_path, ctx_path, assets_path)):
        raise FileNotFoundError(f"Missing telemetry, context, or assets in {data_dir}")

    tel = pd.read_csv(tel_path)
    ctx = pd.read_csv(ctx_path)
    assets = pd.read_csv(assets_path)

    tel["timestamp"] = pd.to_datetime(tel["timestamp"]).dt.tz_localize(None)
    ctx["timestamp"] = pd.to_datetime(ctx["timestamp"]).dt.tz_localize(None)

    # Filter to campus main meters
    meters = assets[assets["asset_type"] == "meter"]["asset_id"].tolist()
    meter_tel = tel[tel["asset_id"].isin(meters)].copy()

    campus_actual = (
        meter_tel.groupby("timestamp")["power_kw"]
        .sum()
        .reset_index()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    campus_weather = ctx.groupby("timestamp")[["outdoor_temperature_c"]].mean().reset_index()
    campus_df = campus_actual.merge(campus_weather, on="timestamp", how="left")
    campus_df["outdoor_temperature_c"] = campus_df["outdoor_temperature_c"].fillna(32.0)

    # Extract holiday dates
    if "holiday" in ctx.columns:
        holiday_mask = ctx["holiday"] == True
        holiday_dates = set(ctx[holiday_mask]["timestamp"].dt.date.tolist())
    else:
        holiday_dates = set()

    return campus_df, holiday_dates


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray | None = None,
    y_upper: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute MAE, MAPE, RMSE, and empirical interval coverage metrics."""
    if len(y_true) == 0:
        return {
            "mae_kw": 0.0,
            "mape_pct": 0.0,
            "rmse_kw": 0.0,
            "coverage_pct": None,
            "count": 0,
        }

    diff = np.abs(y_true - y_pred)
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    valid_mask = y_true > 0.01
    if valid_mask.sum() > 0:
        mape = float(np.mean(diff[valid_mask] / y_true[valid_mask])) * 100.0
    else:
        mape = 0.0

    coverage_pct = None
    if y_lower is not None and y_upper is not None and len(y_lower) == len(y_true):
        covered = (y_true >= y_lower) & (y_true <= y_upper)
        coverage_pct = round(float(np.mean(covered) * 100.0), 2)

    return {
        "mae_kw": round(mae, 2),
        "mape_pct": round(mape, 2),
        "rmse_kw": round(rmse, 2),
        "coverage_pct": coverage_pct,
        "count": int(len(y_true)),
    }


def run_forecast_evaluation(
    data_dir: Path | str | None = None,
    train_days: int = 21,
) -> dict[str, Any]:
    """Execute temporal holdout forecast evaluation across Ridge, Prophet, and Seasonal-Naive."""
    campus_df, holiday_dates = prepare_forecast_data(data_dir)

    min_ts = campus_df["timestamp"].min()
    split_ts = min_ts + timedelta(days=train_days)

    train_df = campus_df[campus_df["timestamp"] < split_ts].copy()
    test_df = campus_df[campus_df["timestamp"] >= split_ts].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(f"Insufficient data for {train_days}-day temporal split: train={len(train_df)}, test={len(test_df)}")

    # Add temporal & calendar features
    for df in (train_df, test_df):
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_holiday"] = df["timestamp"].dt.date.isin(holiday_dates).astype(int)
        df["occupancy"] = 0

    test_df["day_type"] = np.where(
        test_df["is_holiday"] == 1,
        "holiday",
        np.where(test_df["is_weekend"] == 1, "weekend", "weekday"),
    )

    logger.info("Training points: %d (until %s), Test points: %d (from %s)", len(train_df), split_ts, len(test_df), split_ts)

    # ------------------------------------------------------------------ #
    # Model 1: Ridge Regression (ExpectedConsumptionModel baseline)       #
    # ------------------------------------------------------------------ #
    ridge_model = ExpectedConsumptionModel()
    ridge_model.fit(train_df, lower_quantile=10.0, upper_quantile=90.0)
    pred_r, low_r, up_r = ridge_model.predict(
        test_df, return_intervals=True, lower_quantile=10.0, upper_quantile=90.0
    )
    test_df["pred_ridge"] = np.clip(pred_r, 0.0, None)
    test_df["pred_ridge_lower"] = np.clip(low_r, 0.0, None)
    test_df["pred_ridge_upper"] = np.clip(up_r, 0.0, None)

    # ------------------------------------------------------------------ #
    # Model 2: Meta Prophet                                              #
    # ------------------------------------------------------------------ #
    pdf_train = train_df.rename(columns={"timestamp": "ds", "power_kw": "y"})[["ds", "y"]]
    prophet = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_mode="additive",
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
    )
    prophet.fit(pdf_train)
    pdf_test = test_df.rename(columns={"timestamp": "ds"})[["ds"]]
    fc_prophet = prophet.predict(pdf_test)
    pred_p = np.clip(fc_prophet["yhat"].values, 0.0, None)
    low_p = np.clip(fc_prophet["yhat_lower"].values, 0.0, None)
    up_p = np.clip(fc_prophet["yhat_upper"].values, 0.0, None)
    low_p = np.minimum(low_p, pred_p)
    up_p = np.maximum(up_p, pred_p)

    test_df["pred_prophet"] = pred_p
    test_df["pred_prophet_lower"] = low_p
    test_df["pred_prophet_upper"] = up_p

    # ------------------------------------------------------------------ #
    # Model 3: Seasonal-Naive (t - 7 days)                               #
    # ------------------------------------------------------------------ #
    naive_forecaster = SeasonalNaiveForecaster(seasonal_period_days=7, fallback_period_days=1)
    naive_forecaster.fit(train_df, target_col="power_kw", time_col="timestamp")
    fc_naive = naive_forecaster.predict(test_df["timestamp"])
    test_df["pred_naive"] = fc_naive["predicted_power_kw"].values
    test_df["pred_naive_lower"] = None
    test_df["pred_naive_upper"] = None

    # ------------------------------------------------------------------ #
    # Metric Aggregation                                                 #
    # ------------------------------------------------------------------ #
    model_columns = [
        ("Ridge Regression", "pred_ridge", "pred_ridge_lower", "pred_ridge_upper"),
        ("Meta Prophet", "pred_prophet", "pred_prophet_lower", "pred_prophet_upper"),
        ("Seasonal-Naive", "pred_naive", "pred_naive_lower", "pred_naive_upper"),
    ]

    day_types = ["overall", "weekday", "weekend", "holiday"]
    results_by_model: dict[str, dict[str, Any]] = {}

    for model_name, pred_col, low_col, up_col in model_columns:
        model_metrics: dict[str, Any] = {}
        for dt in day_types:
            if dt == "overall":
                sub = test_df
            else:
                sub = test_df[test_df["day_type"] == dt]

            if len(sub) == 0:
                model_metrics[dt] = {
                    "mae_kw": None,
                    "mape_pct": None,
                    "rmse_kw": None,
                    "coverage_pct": None,
                    "count": 0,
                    "note": f"0 {dt} days in test window",
                }
            else:
                y_low = (
                    sub[low_col].values
                    if low_col and low_col in sub.columns and sub[low_col].iloc[0] is not None
                    else None
                )
                y_up = (
                    sub[up_col].values
                    if up_col and up_col in sub.columns and sub[up_col].iloc[0] is not None
                    else None
                )
                model_metrics[dt] = compute_metrics(
                    y_true=sub["power_kw"].values,
                    y_pred=sub[pred_col].values,
                    y_lower=y_low,
                    y_upper=y_up,
                )
        results_by_model[model_name] = model_metrics

    # Produce tabular comparison summary
    comparison_table = []
    for model_name in results_by_model:
        m = results_by_model[model_name]
        cov_val = m["overall"].get("coverage_pct")
        row = {
            "Model": model_name,
            "Overall MAPE (%)": m["overall"]["mape_pct"],
            "Overall MAE (kW)": m["overall"]["mae_kw"],
            "Overall RMSE (kW)": m["overall"]["rmse_kw"],
            "80% Interval Coverage (%)": f"{cov_val:.1f}%" if cov_val is not None else "N/A",
            "Interval Coverage (%)": f"{cov_val:.1f}%" if cov_val is not None else "N/A",
            "Weekday MAPE (%)": m["weekday"]["mape_pct"] if m["weekday"]["count"] > 0 else None,
            "Weekend MAPE (%)": m["weekend"]["mape_pct"] if m["weekend"]["count"] > 0 else None,
            "Holiday MAPE (%)": m["holiday"]["mape_pct"] if m["holiday"]["count"] > 0 else "N/A (0 days)",
        }
        comparison_table.append(row)

    return {
        "split_timestamp": str(split_ts),
        "train_points": len(train_df),
        "test_points": len(test_df),
        "models": results_by_model,
        "comparison_table": comparison_table,
    }
