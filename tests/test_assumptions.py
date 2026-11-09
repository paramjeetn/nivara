"""Unit tests for the Nivara Assumptions Registry."""
from __future__ import annotations

import ast
from pathlib import Path
import pytest

from campus_energy import assumptions
from campus_energy.assumptions import Assumption, AssumptionsRegistry, load_assumptions


def test_load_assumptions_returns_registry():
    """Verify loading config/assumptions.yaml succeeds and yields AssumptionsRegistry."""
    registry = load_assumptions("config/assumptions.yaml")
    assert isinstance(registry, AssumptionsRegistry)
    assert len(registry) > 0
    assert "tariff.electricity_commercial" in registry


def test_get_valid_key():
    """Verify get(key) returns a validated Assumption instance with required fields."""
    item = assumptions.get("tariff.electricity_commercial")
    assert isinstance(item, Assumption)
    assert item.key == "tariff.electricity_commercial"
    assert item.value == 8.5
    assert item.unit == "INR/kWh"
    assert item.confidence == "high"
    assert len(item.source) > 0
    assert len(item.source_date) > 0
    assert "all" in item.applies_to


def test_get_missing_key_raises_key_error():
    """Verify get(missing_key) raises a descriptive KeyError."""
    with pytest.raises(KeyError) as exc_info:
        assumptions.get("non_existent_key_12345")
    assert "non_existent_key_12345" in str(exc_info.value)


def test_get_value_with_and_without_default():
    """Verify get_value returns value, default on missing, or raises KeyError."""
    val = assumptions.get_value("tariff.electricity_commercial")
    assert val == 8.5

    # With default
    fallback = assumptions.get_value("missing_key", default=42.0)
    assert fallback == 42.0

    # Without default
    with pytest.raises(KeyError):
        assumptions.get_value("missing_key")


def test_grid_emission_factor_needs_verification():
    """Verify grid emission factor is explicitly tagged with needs_verification."""
    grid_item = assumptions.get("emissions.grid_factor")
    assert grid_item.value == 0.82
    assert grid_item.unit == "kgCO2e/kWh"
    assert grid_item.confidence == "needs_verification"
    assert "TODO" in (grid_item.notes or "")
    assert "CEA" in grid_item.source


def test_list_by_prefix():
    """Verify prefix search returns all matching assumptions."""
    bee_items = assumptions.list_by_prefix("bee.epi.")
    assert len(bee_items) >= 20  # 4 facility types * (4 star thresholds + 1 ecbc baseline)
    keys = [item.key for item in bee_items]
    assert "bee.epi.university.ecbc_baseline" in keys
    assert "bee.epi.hospital.ecbc_baseline" in keys
    assert "bee.epi.govt_office.ecbc_baseline" in keys
    assert "bee.epi.industrial.ecbc_baseline" in keys


def test_inventory_constants_presence_and_values():
    """Verify all constants cataloged in docs/STATE.md are registered accurately."""
    expected = {
        "tariff.electricity_commercial": 8.5,
        "tariff.water_municipal": 25.0,
        "tariff.tod_peak": 25.0,
        "tariff.tod_mid": 18.0,
        "tariff.tod_off_peak": 10.0,
        "emissions.grid_factor": 0.82,
        "emissions.solar_lifecycle_factor": 0.04,
        "sustainability.tree_sequestration_kg_co2_yr": 21.0,
        "sustainability.coal_displacement_tons_per_mwh": 0.45,
        "policy.napcc_reduction_target_pct": 20.0,
        "waste.fault_avoidable_fraction": 0.80,
        "budget.month_ceiling_days": 30.0,
        "anomaly.statistical.deviation_cutoff_kw": 0.5,
        "anomaly.statistical.pct_deviation_cutoff": 0.3,
        "anomaly.statistical.critical_deviation_kw": 2.0,
        "anomaly.unoccupied.standby_fraction": 0.2,
        "anomaly.pump.dry_run_power_fraction": 0.5,
        "anomaly.ml.isolation_forest_contamination": 0.03,
        "anomaly.alert.dedup_window_hours": 4.0,
        "comfort.optimal_setpoint_c": 24.0,
        "comfort.air_velocity_mps": 0.15,
        "comfort.metabolic_rate_met": 1.1,
        "comfort.clothing_insulation_clo": 0.6,
        "comfort.ambient_fallback_c": 32.0,
        "water.tariff_inr_per_kl": 25.0,
        "water.mnf_start_hour": 1,
        "water.mnf_end_hour": 5,
        "water.leak_spike_multiplier": 3.0,
        "water.leak_absolute_threshold_lph": 500.0,
    }

    for key, expected_val in expected.items():
        actual_val = assumptions.get_value(key)
        assert actual_val == expected_val, f"Key {key}: expected {expected_val}, got {actual_val}"


def test_core_analytics_no_hardcoded_literals_ast():
    """Ensure core analytics files (esg.py, billing.py, carbon_budget.py) contain no 8.5 or 0.82 literals."""
    repo_root = Path(__file__).resolve().parent.parent
    core_files = [
        repo_root / "src" / "campus_energy" / "analytics" / "esg.py",
        repo_root / "src" / "campus_energy" / "analytics" / "billing.py",
        repo_root / "src" / "campus_energy" / "analytics" / "carbon_budget.py",
    ]

    forbidden_literals = {8.5, 0.82}

    for file_path in core_files:
        assert file_path.exists(), f"File not found: {file_path}"
        code = file_path.read_text(encoding="utf-8")

        # 1. AST Constant inspection
        tree = ast.parse(code, filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                val = node.value
                if isinstance(val, (int, float)):
                    for forbidden in forbidden_literals:
                        assert val != forbidden, (
                            f"Hard-coded literal {forbidden} found in {file_path.name} at line {node.lineno}"
                        )

        # 2. String literal inspection (ensuring no hard-coded "8.5" or "0.82" in source)
        for forbidden_str in ["8.5", "0.82"]:
            # Check lines that are not comments or imports
            for idx, line in enumerate(code.splitlines(), start=1):
                clean_line = line.strip()
                if clean_line.startswith("#"):
                    continue
                assert forbidden_str not in clean_line, (
                    f"Forbidden literal '{forbidden_str}' found in {file_path.name} line {idx}: {line}"
                )
