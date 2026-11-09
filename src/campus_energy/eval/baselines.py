"""Baseline models for anomaly detection and load forecasting.

These serve as naive benchmark comparators against which Nivara's
domain-informed layered engines are rigorously evaluated.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Sequence

import numpy as np
import pandas as pd

from campus_energy.schemas import Alert, Severity


def _create_baseline_alert(
    asset_id: str,
    building_id: str,
    zone_id: str | None,
    anomaly_type: str,
    detected_at: datetime | pd.Timestamp,
    event_start: datetime | pd.Timestamp,
    actual_value: float,
    expected_value: float,
    evidence: str,
    recommendation: str,
    severity: Severity = Severity.MEDIUM,
    confidence: float = 0.70,
) -> Alert:
    """Helper to instantiate an Alert schema item for evaluation."""
    dev = actual_value - expected_value
    return Alert(
        alert_id=f"ALT-BASE-{uuid.uuid4().hex[:8].upper()}",
        detected_at=pd.to_datetime(detected_at),
        event_start_estimate=pd.to_datetime(event_start),
        asset_id=asset_id,
        building_id=building_id,
        zone_id=zone_id,
        anomaly_type=anomaly_type,
        severity=severity,
        actual_value=round(float(actual_value), 2),
        expected_value=round(float(expected_value), 2),
        deviation=round(float(dev), 2),
        evidence=evidence,
        recommendation=recommendation,
        confidence=round(float(confidence), 2),
    )


class StaticThresholdDetector:
    """Anomaly Baseline 1: Static threshold detector.

    Flags readings exceeding an absolute power threshold (kW), a historical
    percentile, or a fixed multiplier of the asset's rated capacity.
    """

    def __init__(
        self,
        fixed_threshold_kw: float | None = None,
        percentile: float | None = 95.0,
        rated_power_multiplier: float | None = None,
        min_consecutive: int = 3,
    ) -> None:
        self.fixed_threshold_kw = fixed_threshold_kw
        self.percentile = percentile
        self.rated_power_multiplier = rated_power_multiplier
        self.min_consecutive = min_consecutive

    def detect(
        self,
        features_df: pd.DataFrame,
        assets_df: pd.DataFrame,
    ) -> list[Alert]:
        """Detect anomalies using static upper bound thresholding."""
        alerts: list[Alert] = []
        if features_df.empty or "power_kw" not in features_df.columns:
            return alerts

        asset_ids = features_df["asset_id"].unique()
        assets_map = {row["asset_id"]: row for _, row in assets_df.iterrows()}

        for asset_id in asset_ids:
            asset_info = assets_map.get(asset_id, {})
            building_id = asset_info.get("building_id", "CAMPUS")
            zone_id = asset_info.get("zone_id") if pd.notna(asset_info.get("zone_id")) else None
            rated_power = float(asset_info.get("rated_power_kw", 1.0) or 1.0)

            asset_df = (
                features_df[features_df["asset_id"] == asset_id]
                .copy()
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            if len(asset_df) < self.min_consecutive:
                continue

            # Determine cutoff threshold
            if self.fixed_threshold_kw is not None:
                cutoff = float(self.fixed_threshold_kw)
            elif self.rated_power_multiplier is not None:
                cutoff = rated_power * float(self.rated_power_multiplier)
            elif self.percentile is not None:
                cutoff = float(np.percentile(asset_df["power_kw"].dropna(), self.percentile))
            else:
                cutoff = rated_power * 1.10

            is_breach = (asset_df["power_kw"] > cutoff).astype(int)
            event_id_series = (is_breach.diff() == 1).cumsum()

            for _, group in asset_df[is_breach == 1].groupby(event_id_series):
                if len(group) >= self.min_consecutive:
                    actual_mean = float(group["power_kw"].mean())
                    alerts.append(
                        _create_baseline_alert(
                            asset_id=asset_id,
                            building_id=building_id,
                            zone_id=zone_id,
                            anomaly_type="static_threshold",
                            detected_at=group["timestamp"].max(),
                            event_start=group["timestamp"].min(),
                            actual_value=actual_mean,
                            expected_value=cutoff,
                            evidence=(
                                f"Static threshold breached: power {actual_mean:.2f} kW > "
                                f"cutoff {cutoff:.2f} kW for {len(group)*5} min."
                            ),
                            recommendation="Inspect load against static ceiling constraint.",
                            severity=Severity.HIGH if actual_mean > cutoff * 1.5 else Severity.MEDIUM,
                            confidence=0.75,
                        )
                    )

        return alerts


def detect_static_threshold(
    features_df: pd.DataFrame,
    assets_df: pd.DataFrame,
    fixed_threshold_kw: float | None = None,
    percentile: float | None = 95.0,
    rated_power_multiplier: float | None = None,
    min_consecutive: int = 3,
) -> list[Alert]:
    """Functional wrapper for StaticThresholdDetector."""
    detector = StaticThresholdDetector(
        fixed_threshold_kw=fixed_threshold_kw,
        percentile=percentile,
        rated_power_multiplier=rated_power_multiplier,
        min_consecutive=min_consecutive,
    )
    return detector.detect(features_df, assets_df)


class RollingZScoreDetector:
    """Anomaly Baseline 2: Rolling z-score detector.

    Computes 24h trailing rolling mean and standard deviation (288 steps @ 5-min).
    Flags sustained observations with z-score = (power - mean) / std >= threshold.
    """

    def __init__(
        self,
        window: int = 288,
        z_threshold: float = 3.0,
        min_consecutive: int = 3,
    ) -> None:
        self.window = window
        self.z_threshold = z_threshold
        self.min_consecutive = min_consecutive

    def detect(
        self,
        features_df: pd.DataFrame,
        assets_df: pd.DataFrame,
    ) -> list[Alert]:
        """Detect anomalies using trailing rolling z-score statistics."""
        alerts: list[Alert] = []
        if features_df.empty or "power_kw" not in features_df.columns:
            return alerts

        asset_ids = features_df["asset_id"].unique()
        assets_map = {row["asset_id"]: row for _, row in assets_df.iterrows()}

        for asset_id in asset_ids:
            asset_info = assets_map.get(asset_id, {})
            building_id = asset_info.get("building_id", "CAMPUS")
            zone_id = asset_info.get("zone_id") if pd.notna(asset_info.get("zone_id")) else None

            asset_df = (
                features_df[features_df["asset_id"] == asset_id]
                .copy()
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            if len(asset_df) < max(12, self.min_consecutive):
                continue

            r_mean = asset_df["power_kw"].rolling(self.window, min_periods=12).mean()
            r_std = asset_df["power_kw"].rolling(self.window, min_periods=12).std().fillna(0)

            # Avoid division by zero on static load
            valid_std = r_std > 1e-4
            z_scores = np.where(valid_std, (asset_df["power_kw"] - r_mean) / r_std, 0.0)
            asset_df["z_score"] = z_scores
            asset_df["rolling_mean"] = r_mean

            is_outlier = pd.Series((z_scores >= self.z_threshold).astype(int), index=asset_df.index)
            event_id_series = (is_outlier.diff() == 1).cumsum()

            for _, group in asset_df[is_outlier == 1].groupby(event_id_series):
                if len(group) >= self.min_consecutive:
                    actual_mean = float(group["power_kw"].mean())
                    expected_mean = float(group["rolling_mean"].mean())
                    z_mean = float(group["z_score"].mean())
                    alerts.append(
                        _create_baseline_alert(
                            asset_id=asset_id,
                            building_id=building_id,
                            zone_id=zone_id,
                            anomaly_type="rolling_zscore",
                            detected_at=group["timestamp"].max(),
                            event_start=group["timestamp"].min(),
                            actual_value=actual_mean,
                            expected_value=expected_mean,
                            evidence=(
                                f"Rolling z-score breached: z={z_mean:.2f} >= {self.z_threshold:.1f} "
                                f"(actual {actual_mean:.2f} kW vs 24h rolling {expected_mean:.2f} kW) "
                                f"for {len(group)*5} min."
                            ),
                            recommendation="Investigate rapid deviation from rolling trailing baseline.",
                            severity=Severity.HIGH if z_mean >= 4.5 else Severity.MEDIUM,
                            confidence=min(0.95, 0.60 + z_mean / 10.0),
                        )
                    )

        return alerts


def detect_rolling_zscore(
    features_df: pd.DataFrame,
    assets_df: pd.DataFrame,
    window: int = 288,
    z_threshold: float = 3.0,
    min_consecutive: int = 3,
) -> list[Alert]:
    """Functional wrapper for RollingZScoreDetector."""
    detector = RollingZScoreDetector(
        window=window,
        z_threshold=z_threshold,
        min_consecutive=min_consecutive,
    )
    return detector.detect(features_df, assets_df)


class SeasonalNaiveForecaster:
    """Forecast Baseline: Seasonal-naive forecaster.

    Projects future load using 'same slot last week' (t - 7 days).
    If t - 7 days is unavailable (e.g. at the beginning of series),
    gracefully falls back to 'same slot yesterday' (t - 1 day),
    then to same (hour, minute) historical average, and finally overall mean.
    """

    def __init__(
        self,
        seasonal_period_days: int = 7,
        fallback_period_days: int = 1,
    ) -> None:
        self.seasonal_period_days = seasonal_period_days
        self.fallback_period_days = fallback_period_days
        self._history_map: dict[pd.Timestamp, float] = {}
        self._slot_mean_map: dict[tuple[int, int], float] = {}
        self._global_mean: float = 0.0

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "power_kw",
        time_col: str = "timestamp",
    ) -> "SeasonalNaiveForecaster":
        """Index historical training series for rapid seasonal lookup."""
        clean_df = train_df.copy()
        clean_df[time_col] = pd.to_datetime(clean_df[time_col])
        if clean_df[time_col].dt.tz is not None:
            clean_df[time_col] = clean_df[time_col].dt.tz_localize(None)

        self._history_map = dict(zip(clean_df[time_col], clean_df[target_col].astype(float)))

        clean_df["_hr"] = clean_df[time_col].dt.hour
        clean_df["_min"] = clean_df[time_col].dt.minute
        slot_means = clean_df.groupby(["_hr", "_min"])[target_col].mean().to_dict()
        self._slot_mean_map = {k: float(v) for k, v in slot_means.items()}
        self._global_mean = float(clean_df[target_col].mean()) if len(clean_df) else 0.0
        return self

    def predict(
        self,
        target_timestamps: Sequence[pd.Timestamp | str] | pd.Series,
    ) -> pd.DataFrame:
        """Predict target timestamps using seasonal-naive lookups."""
        ts_series = pd.to_datetime(pd.Series(target_timestamps))
        if ts_series.dt.tz is not None:
            ts_series = ts_series.dt.tz_localize(None)

        preds: list[float] = []
        delta_week = timedelta(days=self.seasonal_period_days)
        delta_day = timedelta(days=self.fallback_period_days)

        for ts in ts_series:
            # 1. Weekly match: same time slot last week
            t_week = ts - delta_week
            if t_week in self._history_map:
                preds.append(self._history_map[t_week])
                continue

            # 2. Daily fallback: same time slot yesterday
            t_day = ts - delta_day
            if t_day in self._history_map:
                preds.append(self._history_map[t_day])
                continue

            # 3. Intraday slot fallback: same hour/minute mean
            slot = (ts.hour, ts.minute)
            if slot in self._slot_mean_map:
                preds.append(self._slot_mean_map[slot])
                continue

            # 4. Global series mean
            preds.append(self._global_mean)

        return pd.DataFrame({
            "timestamp": ts_series,
            "predicted_power_kw": np.clip(np.array(preds, dtype=float), 0.0, None),
        })


def generate_seasonal_naive_forecast(
    train_df: pd.DataFrame,
    target_timestamps: Sequence[pd.Timestamp | str] | pd.Series,
    target_col: str = "power_kw",
    time_col: str = "timestamp",
    seasonal_period_days: int = 7,
) -> pd.DataFrame:
    """Functional wrapper for SeasonalNaiveForecaster."""
    forecaster = SeasonalNaiveForecaster(seasonal_period_days=seasonal_period_days)
    forecaster.fit(train_df, target_col=target_col, time_col=time_col)
    return forecaster.predict(target_timestamps)
