"""Prophet forecaster module: campus-level energy load projection and model comparison."""
import logging
import sys
from datetime import timedelta
import numpy as np
import pandas as pd

# Suppress Prophet / Stan verbose output
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
for _name in ("prophet", "cmdstanpy"):
    _l = logging.getLogger(_name)
    _l.setLevel(logging.WARNING)
    for _h in _l.handlers:
        _h.setLevel(logging.WARNING)

# Workaround for polars environment incompatibility with cmdstanpy CSV reading
# Workaround: polars package in this environment raises RuntimeError (SSE3 flag)
# which breaks cmdstanpy's optional polars loader (it only catches ImportError).
# Intercept 'polars' to raise ImportError cleanly if broken so cmdstanpy falls back to numpy.
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

logger = logging.getLogger(__name__)


def generate_prophet_forecast(
    telemetry_df: pd.DataFrame,
    assets_df: pd.DataFrame,
    context_df: pd.DataFrame | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, float | None]:
    """Fit a Prophet model on campus-level meter load and forecast the next 24 hours.

    Args:
        telemetry_df: Raw or preprocessed telemetry records with timestamp and power_kw.
        assets_df: Asset registry containing asset_id and asset_type.
        context_df: Optional environmental/calendar context DataFrame.
        **kwargs: Additional parameters for signature compatibility.

    Returns:
        tuple containing:
            - forecast_df (pd.DataFrame): 24-hour forecast with columns:
                ['timestamp', 'prophet_predicted_kw', 'prophet_lower_kw', 'prophet_upper_kw']
            - mape (float | None): Mean absolute percentage error on the held-out
              last 24 hours of campus total meter power, or None if evaluation failed.
    """
    empty_df = pd.DataFrame(
        columns=["timestamp", "prophet_predicted_kw", "prophet_lower_kw", "prophet_upper_kw"]
    )

    try:
        # Gracefully handle swapped arguments if called as (tel, ctx, assets)
        if (
            "asset_type" not in assets_df.columns
            and context_df is not None
            and "asset_type" in getattr(context_df, "columns", [])
        ):
            assets_df, context_df = context_df, assets_df

        if telemetry_df is None or assets_df is None or telemetry_df.empty or assets_df.empty:
            return empty_df, None

        tel = telemetry_df.copy()
        tel["timestamp"] = pd.to_datetime(tel["timestamp"])
        if tel["timestamp"].dt.tz is not None:
            tel["timestamp"] = tel["timestamp"].dt.tz_localize(None)

        # Identify campus main meters
        meters = assets_df[assets_df["asset_type"] == "meter"]["asset_id"].tolist()
        meter_tel = tel[tel["asset_id"].isin(meters)].copy()
        if meter_tel.empty:
            return empty_df, None

        # Aggregate campus-level total meter power into 5-minute intervals
        campus_actual = (
            meter_tel.groupby("timestamp")["power_kw"]
            .sum()
            .reset_index()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if len(campus_actual) < 50:
            return empty_df, None

        latest_ts = campus_actual["timestamp"].max()

        # Train on the last 21 days of data
        train_start = latest_ts - timedelta(days=21)
        train_df = campus_actual[campus_actual["timestamp"] >= train_start].copy()
        if len(train_df) < 50:
            return empty_df, None

        # ── Held-out evaluation on the last 24 hours ──
        mape: float | None = None
        eval_cutoff = latest_ts - timedelta(days=1)
        eval_train = campus_actual[
            (campus_actual["timestamp"] >= train_start) & (campus_actual["timestamp"] <= eval_cutoff)
        ].copy()
        eval_test = campus_actual[campus_actual["timestamp"] > eval_cutoff].copy()

        if len(eval_train) >= 50 and len(eval_test) >= 10:
            try:
                pdf_eval = eval_train.rename(columns={"timestamp": "ds", "power_kw": "y"})
                m_eval = Prophet(
                    changepoint_prior_scale=0.05,
                    seasonality_mode="additive",
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                )
                m_eval.fit(pdf_eval)
                future_eval = m_eval.make_future_dataframe(
                    periods=len(eval_test), freq="5min", include_history=False
                )
                fc_eval = m_eval.predict(future_eval)
                pred_eval_vals = np.clip(fc_eval["yhat"].values, 0, None)
                actual_eval_vals = eval_test["power_kw"].values

                # Compute MAPE vs held-out actuals
                valid_mask = actual_eval_vals > 0.01
                if valid_mask.sum() > 0:
                    mape = float(
                        np.mean(
                            np.abs(
                                (actual_eval_vals[valid_mask] - pred_eval_vals[valid_mask])
                                / actual_eval_vals[valid_mask]
                            )
                        )
                    )
            except Exception as eval_err:
                logger.warning("Prophet held-out evaluation failed: %s", eval_err)
                mape = None

        # ── Fit Prophet model on the full 21 days for forward 24h forecast ──
        pdf_full = train_df.rename(columns={"timestamp": "ds", "power_kw": "y"})
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_mode="additive",
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
        )
        model.fit(pdf_full)

        # Forecast next 24 hours (288 x 5-minute points)
        future_df = model.make_future_dataframe(periods=288, freq="5min", include_history=False)
        forecast = model.predict(future_df)

        result_df = pd.DataFrame(
            {
                "timestamp": forecast["ds"],
                "prophet_predicted_kw": np.clip(forecast["yhat"], 0, None),
                "prophet_lower_kw": np.clip(forecast["yhat_lower"], 0, None),
                "prophet_upper_kw": np.clip(forecast["yhat_upper"], 0, None),
            }
        )

        return result_df, mape

    except Exception as err:
        logger.warning("generate_prophet_forecast failed: %s", err)
        return empty_df, None
