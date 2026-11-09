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
    "RecommendationEngine",
    "generate_recommendations_from_alerts",
    "generate_recommendations_from_forecast",
    "generate_recommendations_from_scenarios",
    "generate_recommendations_from_complaints",
    "build_ranked_recommendations",
]
