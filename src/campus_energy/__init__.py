'''Campus Energy Intelligence Platform.'''
from campus_energy import assumptions
from campus_energy import profiles
from campus_energy import decision
from campus_energy import tariffs
from campus_energy import notify
from campus_energy import roles
from campus_energy import demo
from campus_energy import adapters
from campus_energy.demo import (
    GuidedTour,
    inject_live_fault,
    active_fault_registry,
    clear_fault_registry,
)
from campus_energy.roles import (
    RoleDefinition,
    RolesRegistry,
    load_roles,
    get_role,
)
from campus_energy.notify import (
    BaseNotifier,
    EmailNotifier,
    NotificationMessage,
    NotificationResult,
    NotificationRouter,
    TelegramNotifier,
    get_notification_router,
)

__all__ = [
    "assumptions",
    "profiles",
    "decision",
    "tariffs",
    "notify",
    "roles",
    "demo",
    "adapters",
    "GuidedTour",
    "inject_live_fault",
    "active_fault_registry",
    "clear_fault_registry",
    "RoleDefinition",
    "RolesRegistry",
    "load_roles",
    "get_role",
    "BaseNotifier",
    "EmailNotifier",
    "NotificationMessage",
    "NotificationResult",
    "NotificationRouter",
    "TelegramNotifier",
    "get_notification_router",
]
