"""Governance, policy compliance, and multi-facility benchmarking analytics."""
from pathlib import Path
import re
from typing import Dict, Any
import pandas as pd

from campus_energy import assumptions
from campus_energy.analytics.esg import calculate_esg_metrics

GRID_EMISSION_FACTOR_KG_PER_KWH: float = assumptions.get_value("emissions.grid_factor")


def clean_facility_label(label: str) -> str:
    """Format human-readable facility label by stripping verbose suffixes."""
    # Split on em-dash or regular dash if present
    name = label.split("—")[0].strip() if "—" in label else label.split("-")[0].strip()
    # Strip any leading emojis and whitespace
    clean = re.sub(r"^[^\w\s]+", "", name).strip()
    return clean if clean else label.strip()


def get_multi_facility_benchmarks(facility_options: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate and benchmark performance across multiple institutional facilities.

    Args:
        facility_options: Dictionary mapping facility display names to configuration metadata:
            {
                "key": facility_type_str,
                "data_dir": Path or str,
                "floor_area_m2": float or int,
                "peak_occupancy": int,
                ...
            }

    Returns:
        pd.DataFrame containing standardized governance benchmarks across all facilities.
    """
    rows = []

    for label, meta in facility_options.items():
        data_dir = Path(meta.get("data_dir", "data/generated"))
        facility_type = meta.get("key", "university")
        floor_area_m2 = float(meta.get("floor_area_m2", 10000.0))
        peak_occupancy = int(meta.get("peak_occupancy", 1000))
        display_name = clean_facility_label(label)

        tel_path = data_dir / "telemetry.csv"
        evt_path = data_dir / "events.csv"

        if not tel_path.exists() or not evt_path.exists():
            # Graceful fallback if facility dataset not yet generated
            rows.append({
                "Facility": display_name,
                "Facility Type": facility_type,
                "Floor Area (m²)": floor_area_m2,
                "Peak Occupancy": peak_occupancy,
                "Annual Energy (MWh)": 0.0,
                "Annual Cost (₹ Lakh)": 0.0,
                "EPI (kWh/m²/yr)": 0.0,
                "ECBC Baseline": 0.0,
                "BEE Rating": "N/A",
                "ECBC Status": "Pending Data",
                "Annual Carbon (t CO₂)": 0.0,
                "CO₂ / Occupant (t/person)": 0.0,
                "Avoidable Waste (MWh)": 0.0,
                "Potential Savings (₹ Lakh)": 0.0,
                "Avoidable Carbon (t CO₂)": 0.0,
            })
            continue

        tel = pd.read_csv(tel_path)
        events = pd.read_csv(evt_path)

        res = calculate_esg_metrics(
            telemetry_df=tel,
            events_df=events,
            floor_area_m2=floor_area_m2,
            facility_type=facility_type,
        )

        kpis = res["kpis"]
        bee = res["bee"]

        annual_kwh = float(kpis.get("annual_kwh_projection", 0.0))
        annual_cost_inr = float(kpis.get("annual_cost_inr_projection", 0.0))
        annual_carbon_tons = float(kpis.get("annual_carbon_tons_projection", 0.0))
        wasted_kwh = float(kpis.get("wasted_kwh", 0.0))
        avoidable_cost_inr = float(kpis.get("avoidable_cost_inr", 0.0))
        avoidable_carbon_kg = float(kpis.get("avoidable_carbon_kg", 0.0))

        stars = int(bee.get("stars", 1))
        bee_rating_str = f"{'⭐' * stars} ({stars}/5)"
        ecbc_status = "Compliant" if bee.get("compliant", False) else "Non-Compliant"
        co2_per_occupant = round(annual_carbon_tons / max(peak_occupancy, 1), 2)

        rows.append({
            "Facility": display_name,
            "Facility Type": facility_type,
            "Floor Area (m²)": floor_area_m2,
            "Peak Occupancy": peak_occupancy,
            "Annual Energy (MWh)": round(annual_kwh / 1000, 1),
            "Annual Cost (₹ Lakh)": round(annual_cost_inr / 100_000, 2),
            "EPI (kWh/m²/yr)": float(bee.get("epi_annual_kwh_m2", 0.0)),
            "ECBC Baseline": float(bee.get("ecbc_baseline_kwh_m2", 0.0)),
            "BEE Rating": bee_rating_str,
            "ECBC Status": ecbc_status,
            "Annual Carbon (t CO₂)": round(annual_carbon_tons, 2),
            "CO₂ / Occupant (t/person)": co2_per_occupant,
            "Avoidable Waste (MWh)": round(wasted_kwh / 1000, 1),
            "Potential Savings (₹ Lakh)": round(avoidable_cost_inr / 100_000, 2),
            "Avoidable Carbon (t CO₂)": round(avoidable_carbon_kg / 1000, 2),
        })

    return pd.DataFrame(rows)


def calculate_state_impact(benchmark_df: pd.DataFrame, num_facilities: int = 100) -> Dict[str, Any]:
    """Calculate state-level scaling impact if Nivara is deployed across multiple facilities.

    Args:
        benchmark_df: Summary DataFrame from get_multi_facility_benchmarks.
        num_facilities: Number of institutions/facilities to scale across (default: 100).

    Returns:
        Dictionary with aggregated annual energy, financial, and environmental metrics.
    """
    if benchmark_df.empty or num_facilities <= 0:
        return {
            "total_annual_mwh_saved": 0.0,
            "total_financial_savings_cr": 0.0,
            "total_co2_mitigated_tons": 0.0,
            "trees_planted_equivalent": 0,
            "coal_unburned_tons": 0.0,
        }

    # Filter out empty or placeholder rows if any
    valid_df = benchmark_df[benchmark_df["Annual Energy (MWh)"] > 0]
    if valid_df.empty:
        valid_df = benchmark_df

    avg_avoidable_mwh = float(valid_df["Avoidable Waste (MWh)"].mean())
    avg_avoidable_cost_lakh = float(valid_df["Potential Savings (₹ Lakh)"].mean())
    avg_avoidable_cost_inr = avg_avoidable_cost_lakh * 100_000

    if "Avoidable Carbon (t CO₂)" in valid_df.columns:
        avg_avoidable_carbon_tons = float(valid_df["Avoidable Carbon (t CO₂)"].mean())
    else:
        avg_avoidable_carbon_tons = avg_avoidable_mwh * GRID_EMISSION_FACTOR_KG_PER_KWH

    total_annual_mwh_saved = round(avg_avoidable_mwh * num_facilities, 1)
    total_financial_savings_cr = round((avg_avoidable_cost_inr * num_facilities) / 10_000_000, 2)
    total_co2_mitigated_tons = round(avg_avoidable_carbon_tons * num_facilities, 1)
    tree_seq_kg = assumptions.get_value("sustainability.tree_sequestration_kg_co2_yr")
    trees_planted_equivalent = int(total_co2_mitigated_tons * 1000 / tree_seq_kg)
    coal_factor = assumptions.get_value("sustainability.coal_displacement_tons_per_mwh")
    coal_unburned_tons = round(total_annual_mwh_saved * coal_factor, 1)

    return {
        "total_annual_mwh_saved": total_annual_mwh_saved,
        "total_financial_savings_cr": total_financial_savings_cr,
        "total_co2_mitigated_tons": total_co2_mitigated_tons,
        "trees_planted_equivalent": trees_planted_equivalent,
        "coal_unburned_tons": coal_unburned_tons,
    }
