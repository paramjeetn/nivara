"""Nivara Operational Views Package."""
from campus_energy.ui.views.alerts_work import render_alerts_work
from campus_energy.ui.views.assets import create_2d_campus_map_figure, render_assets
from campus_energy.ui.views.energy import render_energy
from campus_energy.ui.views.home_admin import render_home_admin
from campus_energy.ui.views.home_operations import render_home_operations
from campus_energy.ui.views.home_sustainability import render_home_sustainability
from campus_energy.ui.views.water import render_water

__all__ = [
    "create_2d_campus_map_figure",
    "render_alerts_work",
    "render_assets",
    "render_energy",
    "render_home_admin",
    "render_home_operations",
    "render_home_sustainability",
    "render_water",
]
