"""Base data models and abstract notifier interface for Nivara notifications."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Literal, Optional
from pydantic import BaseModel, Field

RecipientRole = Literal["operations", "administrator", "sustainability", "all"]
SeverityLevel = Literal["critical", "high", "medium", "low"]


class NotificationMessage(BaseModel):
    """Strongly typed notification payload for facility alerts and work orders."""

    recipient_role: RecipientRole = Field(
        default="operations",
        description="Target institutional stakeholder role ('operations', 'administrator', 'sustainability', 'all')",
    )
    severity: SeverityLevel = Field(
        default="high",
        description="Urgency/severity level of the alert ('critical', 'high', 'medium', 'low')",
    )
    title: str = Field(
        description="Concise, plain-language notification subject or headline",
    )
    asset_id: str = Field(
        description="Target asset identifier (e.g. 'CHILLER-01', 'SUBMETER-BLDG-B')",
    )
    description: str = Field(
        description="Actionable, plain-language explanation of anomaly or work order",
    )
    impact_range: str = Field(
        description="Three-point financial impact range (e.g. '₹2,400 [₹1,920 – ₹2,880] / day')",
    )
    service_guard_note: str = Field(
        description="Service-Impact Guard assessment (e.g. '🟢 Service-Safe: Non-critical off-hours load')",
    )
    work_order_id: Optional[str] = Field(
        default=None,
        description="Associated maintenance work order identifier if dispatched",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when notification was generated",
    )


class NotificationResult(BaseModel):
    """Result status of a notification dispatch attempt across a channel."""

    success: bool = Field(description="True if delivered or successfully simulated in dry-run")
    channel: str = Field(description="Delivery channel name (e.g. 'telegram', 'email', 'router')")
    dry_run: bool = Field(description="True if dispatched in simulation/dry-run mode without external network calls")
    message_id: Optional[str] = Field(default=None, description="External provider or simulation message ID")
    error: Optional[str] = Field(default=None, description="Error message if delivery failed")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Delivery timestamp")


class BaseNotifier(ABC):
    """Abstract base class for all notification channel delivery providers."""

    @property
    @abstractmethod
    def channel_name(self) -> str:
        """Channel identifier string (e.g. 'telegram', 'email')."""
        pass

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """Return True if real outbound network credentials are configured."""
        pass

    @abstractmethod
    def send(self, message: NotificationMessage) -> NotificationResult:
        """Dispatch a notification message through this channel."""
        pass
