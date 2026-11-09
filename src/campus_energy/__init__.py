'''Campus Energy Intelligence Platform.'''
from campus_energy import assumptions
from campus_energy import profiles
from campus_energy import decision
from campus_energy import tariffs
from campus_energy import notify
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
    "BaseNotifier",
    "EmailNotifier",
    "NotificationMessage",
    "NotificationResult",
    "NotificationRouter",
    "TelegramNotifier",
    "get_notification_router",
]
