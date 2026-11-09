"""Decision support, Service-Impact Guard, and Unified Recommendation Engine for Nivara."""
from campus_energy.decision.recommendation import (
    GuardResult,
    Recommendation,
    SavingsBundle,
    SavingsRange,
)
from campus_energy.decision.guard import (
    apply_service_guard,
    check_service_guard,
)
from campus_energy.decision.savings import (
    calculate_savings,
    get_benchmark_capital_cost,
    get_savings_assumption_keys,
)
from campus_energy.decision.tariffs import (
    DEFAULT_TARIFF_ID,
    IndianTariff,
    ToDBand,
    list_available_tariffs,
    load_tariff,
)
from campus_energy.decision.india_levers import (
    audit_diesel_generators,
    calculate_apfc_requirements,
    calculate_telemetry_kva,
    generate_dg_audit_recommendation,
    generate_india_tariff_recommendations,
    generate_peak_demand_recommendation,
    generate_power_factor_recommendation,
    generate_tod_load_shifting_recommendations,
)
from campus_energy.decision.engine import (
    RecommendationEngine,
    build_ranked_recommendations,
    generate_recommendations_from_alerts,
    generate_recommendations_from_complaints,
    generate_recommendations_from_forecast,
    generate_recommendations_from_scenarios,
)

__all__ = [
    "SavingsRange",
    "SavingsBundle",
    "GuardResult",
    "Recommendation",
    "check_service_guard",
    "apply_service_guard",
    "calculate_savings",
    "get_benchmark_capital_cost",
    "get_savings_assumption_keys",
    "DEFAULT_TARIFF_ID",
    "IndianTariff",
    "ToDBand",
    "load_tariff",
    "list_available_tariffs",
    "audit_diesel_generators",
    "calculate_apfc_requirements",
    "calculate_telemetry_kva",
    "generate_dg_audit_recommendation",
    "generate_india_tariff_recommendations",
    "generate_peak_demand_recommendation",
    "generate_power_factor_recommendation",
    "generate_tod_load_shifting_recommendations",
    "RecommendationEngine",
    "generate_recommendations_from_alerts",
    "generate_recommendations_from_forecast",
    "generate_recommendations_from_scenarios",
    "generate_recommendations_from_complaints",
    "build_ranked_recommendations",
]
