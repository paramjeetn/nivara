"""Notification subsystem for Nivara Campus Energy Intelligence Platform.

Provides multi-channel notification dispatch (Telegram, Email/SMTP) with role-based routing,
suppression window de-duplication, Service-Impact Guard badges, and IPMVP Option B savings context.
"""
from campus_energy.notify.base import (
    BaseNotifier,
    NotificationMessage,
    NotificationResult,
    RecipientRole,
    SeverityLevel,
)
from campus_energy.notify.email import EmailNotifier
from campus_energy.notify.router import NotificationRouter, get_notification_router
from campus_energy.notify.telegram import TelegramNotifier

__all__ = [
    "BaseNotifier",
    "EmailNotifier",
    "NotificationMessage",
    "NotificationResult",
    "NotificationRouter",
    "RecipientRole",
    "SeverityLevel",
    "TelegramNotifier",
    "get_notification_router",
]
