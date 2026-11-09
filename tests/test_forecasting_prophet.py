"""Unit tests for the Prophet forecasting module."""
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from campus_energy.analytics.forecasting_prophet import generate_prophet_forecast

DATA_DIR = Path(__file__).parent.parent / "data" / "generated"


EXPECTED_COLS = [
    "timestamp",
    "prophet_predicted_kw",
    "prophet_lower_kw",
    "prophet_upper_kw",
    "yhat",
    "yhat_lower",
    "yhat_upper",
]


def test_prophet_forecast_empty_inputs():
    """Verify safe fallback on empty DataFrames."""
    df, mape = generate_prophet_forecast(pd.DataFrame(), pd.DataFrame())
    assert df.empty
    assert list(df.columns) == EXPECTED_COLS
    assert mape is None


def test_prophet_forecast_synthetic():
    """Test Prophet forecast with synthetic 5-minute data over 21 days."""
    dates = pd.date_range("2025-01-01", periods=288 * 22, freq="5min")
    power = 30.0 + 10.0 * np.sin(np.linspace(0, 44 * np.pi, len(dates)))
    tel_df = pd.DataFrame({
        "timestamp": dates,
        "asset_id": "METER-01",
        "power_kw": power,
    })
    assets_df = pd.DataFrame({
        "asset_id": ["METER-01"],
        "asset_type": ["meter"],
    })

    df, mape = generate_prophet_forecast(tel_df, assets_df)

    assert not df.empty
    assert len(df) == 288
    assert list(df.columns) == EXPECTED_COLS
    assert (df["prophet_predicted_kw"] >= 0).all()
    assert (df["prophet_lower_kw"] >= 0).all()
    assert (df["prophet_upper_kw"] >= df["prophet_lower_kw"]).all()
    assert (df["yhat_lower"] <= df["yhat"]).all()
    assert (df["yhat"] <= df["yhat_upper"]).all()
    assert mape is not None
    assert isinstance(mape, float)
    assert 0.0 <= mape < 1.0


def test_prophet_forecast_with_generated_data():
    """Test Prophet forecast using generated campus dataset if available."""
    tel_path = DATA_DIR / "telemetry.csv"
    assets_path = DATA_DIR / "assets.csv"

    if not tel_path.exists() or not assets_path.exists():
        pytest.skip("Generated dataset not found.")

    tel = pd.read_csv(tel_path)
    assets = pd.read_csv(assets_path)

    df, mape = generate_prophet_forecast(tel, assets)

    assert not df.empty
    assert len(df) == 288
    assert list(df.columns) == EXPECTED_COLS
    assert mape is not None
    assert isinstance(mape, float)
