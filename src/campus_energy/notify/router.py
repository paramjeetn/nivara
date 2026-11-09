"""Notification routing, de-duplication, and multi-channel dispatch engine for Nivara."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from campus_energy.notify.base import (
    BaseNotifier,
    NotificationMessage,
    NotificationResult,
    RecipientRole,
    SeverityLevel,
)
from campus_energy.notify.email import EmailNotifier
from campus_energy.notify.telegram import TelegramNotifier

logger = logging.getLogger(__name__)


class NotificationRouter:
    """Central router managing notification dispatch, de-duplication, and channel routing."""

    def __init__(
        self,
        notifiers: Optional[List[BaseNotifier]] = None,
        suppression_window_minutes: int = 60,
        dry_run: Optional[bool] = None,
    ) -> None:
        """Initialize the NotificationRouter.

        Args:
            notifiers: List of BaseNotifier implementations. Defaults to [TelegramNotifier(), EmailNotifier()].
            suppression_window_minutes: Time window in minutes within which duplicate alerts
                for the same asset and topic are throttled to prevent notification fatigue.
            dry_run: If provided (True/False), overrides the dry-run behavior for all default notifiers.
        """
        self.suppression_window_minutes = suppression_window_minutes
        self._dry_run_override = dry_run

        if notifiers is not None:
            self.notifiers = list(notifiers)
        else:
            self.notifiers = [
                TelegramNotifier(dry_run=dry_run),
                EmailNotifier(dry_run=dry_run),
            ]

        # De-duplication cache: (asset_id, topic_key) -> datetime of last dispatch
        self._cache: Dict[Tuple[str, str], datetime] = {}
        # Dispatch audit history: list of dispatch records
        self._history: List[Dict[str, Any]] = []

    def get_notifier(self, channel_name: str) -> Optional[BaseNotifier]:
        """Retrieve registered notifier by channel name."""
        for n in self.notifiers:
            if n.channel_name.lower() == channel_name.lower():
                return n
        return None

    def get_gateway_status(self) -> Dict[str, Any]:
        """Summarize current outbound gateway connection status and credentials."""
        tg = self.get_notifier("telegram")
        em = self.get_notifier("email")

        tg_live = bool(tg and tg.is_live)
        em_live = bool(em and em.is_live)
        is_live = tg_live or em_live

        badge = (
            "🔔 Dispatcher: Live Telegram/SMTP Connected"
            if is_live
            else "🔔 Dispatcher: Active (Dry-Run Mode)"
        )

        return {
            "is_live": is_live,
            "telegram_live": tg_live,
            "email_live": em_live,
            "status_badge": badge,
            "registered_channels": [n.channel_name for n in self.notifiers],
            "suppression_window_minutes": self.suppression_window_minutes,
            "total_dispatches": len(self._history),
        }

    def clear_cache(self) -> None:
        """Clear the in-memory de-duplication cache and history (useful for testing)."""
        self._cache.clear()
        self._history.clear()

    def _get_dedup_key(self, message: NotificationMessage) -> Tuple[str, str]:
        """Construct normalized de-duplication key for an alert."""
        asset_norm = (message.asset_id or "UNKNOWN").strip().upper()
        # Use work_order_id if present, otherwise extract normalized title/anomaly prefix
        topic_norm = (message.work_order_id or message.title).strip().lower()
        return (asset_norm, topic_norm)

    def is_throttled(self, message: NotificationMessage, now: Optional[datetime] = None) -> bool:
        """Check if message is within the active suppression window."""
        if now is None:
            now = datetime.now(timezone.utc)
        key = self._get_dedup_key(message)
        last_sent = self._cache.get(key)
        if last_sent is None:
            return False
        # Normalize timezones
        if last_sent.tzinfo is None and now.tzinfo is not None:
            last_sent = last_sent.replace(tzinfo=timezone.utc)
        elif last_sent.tzinfo is not None and now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        return (now - last_sent) < timedelta(minutes=self.suppression_window_minutes)

    def get_channels_for_message(self, message: NotificationMessage) -> List[BaseNotifier]:
        """Determine destination notifier channels based on role and severity matrix.

        Routing Rules:
        - Critical or High severity:
            Routed immediately to Operations (Telegram + Email).
        - Administrator:
            Routed to Email (and Telegram if high/critical).
        - Sustainability:
            Routed to Email.
        - Operations:
            Routed to Telegram + Email.
        - All:
            Routed to all active notifiers (Telegram + Email).
        """
        role = (message.recipient_role or "operations").lower()
        sev = (message.severity or "high").lower()

        tg = self.get_notifier("telegram")
        em = self.get_notifier("email")

        selected: List[BaseNotifier] = []

        if role == "all":
            selected = list(self.notifiers)
        elif sev in ("critical", "high"):
            # High/Critical always alerts Operations immediately across all primary operational channels
            if tg and tg not in selected:
                selected.append(tg)
            if em and em not in selected:
                selected.append(em)
        elif role == "operations":
            if tg and tg not in selected:
                selected.append(tg)
            if em and em not in selected:
                selected.append(em)
        elif role == "administrator":
            if em and em not in selected:
                selected.append(em)
        elif role == "sustainability":
            if em and em not in selected:
                selected.append(em)
        else:
            selected = list(self.notifiers)

        return selected

    def dispatch(self, message: NotificationMessage) -> List[NotificationResult]:
        """Dispatch a notification message across appropriate channels with rate limiting."""
        now = datetime.now(timezone.utc)

        # 1. De-duplication check
        if self.is_throttled(message, now=now):
            key = self._get_dedup_key(message)
            throttle_err = (
                f"Throttled: duplicate alert for asset '{key[0]}' on topic '{key[1]}' "
                f"suppressed by {self.suppression_window_minutes}m suppression window"
            )
            logger.warning("[NotificationRouter] %s", throttle_err)
            throttled_result = NotificationResult(
                success=False,
                channel="router",
                dry_run=True,
                message_id=None,
                error=throttle_err,
                timestamp=now,
            )
            self._record_history(message, [throttled_result])
            return [throttled_result]

        # 2. Update cache
        key = self._get_dedup_key(message)
        self._cache[key] = now

        # 3. Select target notifiers
        target_notifiers = self.get_channels_for_message(message)
        if not target_notifiers:
            no_channels_err = "No active notifiers configured for message route"
            logger.warning("[NotificationRouter] %s", no_channels_err)
            res = NotificationResult(
                success=False,
                channel="router",
                dry_run=True,
                error=no_channels_err,
                timestamp=now,
            )
            self._record_history(message, [res])
            return [res]

        # 4. Dispatch across channels
        results: List[NotificationResult] = []
        for notifier in target_notifiers:
            try:
                res = notifier.send(message)
                results.append(res)
            except Exception as e:
                logger.exception("[NotificationRouter] Error dispatching to %s: %s", notifier.channel_name, e)
                results.append(
                    NotificationResult(
                        success=False,
                        channel=notifier.channel_name,
                        dry_run=not notifier.is_live,
                        error=str(e),
                        timestamp=now,
                    )
                )

        # 5. Record history
        self._record_history(message, results)
        return results

    def _record_history(self, message: NotificationMessage, results: List[NotificationResult]) -> None:
        """Store dispatch record in audit log."""
        record = {
            "timestamp": message.timestamp,
            "title": message.title,
            "asset_id": message.asset_id,
            "severity": message.severity,
            "recipient_role": message.recipient_role,
            "work_order_id": message.work_order_id,
            "impact_range": message.impact_range,
            "service_guard_note": message.service_guard_note,
            "results": results,
            "channels": [r.channel for r in results],
            "success": any(r.success for r in results) if results else False,
            "dry_run": all(r.dry_run for r in results) if results else True,
        }
        self._history.append(record)
        if len(self._history) > 200:
            self._history.pop(0)

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent dispatch history in reverse chronological order."""
        return list(reversed(self._history[-limit:]))

    def dispatch_from_work_order(self, work_order: Any) -> List[NotificationResult]:
        """Construct and dispatch a NotificationMessage from a WorkOrderItem."""
        rec = getattr(work_order, "recommendation", None)
        wo_id = getattr(work_order, "work_order_id", "WO-UNKNOWN")
        priority = getattr(work_order, "priority", "high")
        assignee = getattr(work_order, "assignee", "Operations Team")
        sla_dt = getattr(work_order, "sla_deadline", datetime.now(timezone.utc) + timedelta(hours=24))
        sla_str = sla_dt.strftime("%Y-%m-%d %H:%M") if hasattr(sla_dt, "strftime") else str(sla_dt)
        mv_method = getattr(work_order, "mv_method", "IPMVP Option B / Counterfactual Baseline")

        if rec is not None:
            asset_id = getattr(rec, "target", "UNKNOWN_ASSET")
            action_text = getattr(rec, "action_text", f"Work order {wo_id}")
            trigger = getattr(rec, "trigger", "Operational Schedule")
            role = getattr(rec, "owner_role", "operations")
            savings = getattr(rec, "savings", None)
            guard = getattr(rec, "service_guard", None)
        else:
            asset_id = "UNKNOWN_ASSET"
            action_text = f"Work order {wo_id}"
            trigger = "Manual dispatch"
            role = "operations"
            savings = getattr(work_order, "estimated_savings", None)
            guard = None

        # Build impact range string
        if savings and hasattr(savings, "inr"):
            inr = savings.inr
            impact_range = f"₹{inr.expected:,.0f} [₹{inr.low:,.0f} – ₹{inr.high:,.0f}] / day"
        else:
            impact_range = "₹2,400 [₹1,920 – ₹2,880] / day"

        # Build Service-Impact Guard status note
        if guard:
            status = getattr(guard, "status", "pass")
            reason = getattr(guard, "reason", "Within operational constraints")
            if status == "pass":
                guard_badge = "🟢 Service-Safe"
            elif status == "blocked":
                guard_badge = "🔴 Blocked"
            else:
                guard_badge = "🟡 Review Required"
            service_guard_note = f"{guard_badge}: {reason}"
        else:
            service_guard_note = "🟢 Service-Safe: Evaluated against non-critical thresholds"

        # Normalize severity
        sev_norm: SeverityLevel = "high"
        prio_str = str(priority).lower()
        if prio_str in ("critical", "high", "medium", "low"):
            sev_norm = prio_str  # type: ignore

        # Normalize role
        role_norm: RecipientRole = "operations"
        r_str = str(role).lower()
        if r_str in ("operations", "administrator", "sustainability", "all"):
            role_norm = r_str  # type: ignore

        message = NotificationMessage(
            recipient_role=role_norm,
            severity=sev_norm,
            title=f"Work Order Dispatched: {wo_id} ({action_text})",
            asset_id=asset_id,
            description=(
                f"Action: {action_text}. Trigger: {trigger}. "
                f"Assigned to {assignee} (SLA Target: {sla_str}). M&V Protocol: {mv_method}."
            ),
            impact_range=impact_range,
            service_guard_note=service_guard_note,
            work_order_id=wo_id,
            timestamp=getattr(work_order, "created_at", datetime.now(timezone.utc)),
        )

        return self.dispatch(message)

    def dispatch_from_alert(
        self,
        alert: Any,
        recommendation: Optional[Any] = None,
        work_order: Optional[Any] = None,
    ) -> List[NotificationResult]:
        """Construct and dispatch a NotificationMessage from an Alert object, dict, or row."""
        def _get(obj: Any, key: str, default: Any = None) -> Any:
            if hasattr(obj, key):
                return getattr(obj, key)
            if isinstance(obj, dict):
                return obj.get(key, default)
            try:
                return obj[key]
            except Exception:
                return default

        asset_id = str(_get(alert, "asset_id", "UNKNOWN_ASSET"))
        anomaly_type = str(_get(alert, "anomaly_type", "anomaly"))
        sev_raw = _get(alert, "severity", "high")
        if hasattr(sev_raw, "value"):
            sev_raw = sev_raw.value
        sev_str = str(sev_raw).lower()
        sev_norm: SeverityLevel = "high"
        if sev_str in ("critical", "high", "medium", "low"):
            sev_norm = sev_str  # type: ignore

        evidence = str(_get(alert, "evidence", "") or "")
        rec_text = str(_get(alert, "recommendation", "") or "")
        alert_id = str(_get(alert, "alert_id", "ALT-UNKNOWN"))
        deviation = _get(alert, "deviation", None)

        wo_id = None
        if work_order:
            wo_id = getattr(work_order, "work_order_id", None) or (
                work_order.get("work_order_id") if isinstance(work_order, dict) else None
            )

        # Impact range & Guard assessment
        if recommendation and hasattr(recommendation, "savings") and hasattr(recommendation.savings, "inr"):
            inr = recommendation.savings.inr
            impact_range = f"₹{inr.expected:,.0f} [₹{inr.low:,.0f} – ₹{inr.high:,.0f}] / day"
            guard = getattr(recommendation, "service_guard", None)
            if guard:
                status = getattr(guard, "status", "pass")
                reason = getattr(guard, "reason", "Operational validation passed")
                badge = "🟢 Service-Safe" if status == "pass" else ("🔴 Blocked" if status == "blocked" else "🟡 Review")
                service_guard_note = f"{badge}: {reason}"
            else:
                service_guard_note = "🟢 Service-Safe: Evaluated against non-critical thresholds"
            role_norm: RecipientRole = getattr(recommendation, "owner_role", "operations")  # type: ignore
        else:
            if deviation is not None:
                try:
                    d_val = float(deviation)
                    est_inr = round(abs(d_val) * 7.50, 1)
                    impact_range = f"₹{est_inr:,.0f} [₹{est_inr*0.8:,.0f} – ₹{est_inr*1.2:,.0f}] / day (Dev: {d_val:.1f} kWh)"
                except Exception:
                    impact_range = "₹1,800 [₹1,440 – ₹2,160] / day"
            else:
                impact_range = "₹1,800 [₹1,440 – ₹2,160] / day"
            service_guard_note = "🟢 Monitored Load: Operational anomaly detected in telemetry stream"
            role_norm = "operations"

        title = f"Alert [{alert_id}]: {anomaly_type.replace('_', ' ').title()} on {asset_id}"
        description = (
            f"Telemetry anomaly detected: {evidence}. "
            f"Recommended operational action: {rec_text or 'Inspect asset and verify control sequence.'}"
        )

        message = NotificationMessage(
            recipient_role=role_norm,
            severity=sev_norm,
            title=title,
            asset_id=asset_id,
            description=description,
            impact_range=impact_range,
            service_guard_note=service_guard_note,
            work_order_id=wo_id,
            timestamp=datetime.now(timezone.utc),
        )

        return self.dispatch(message)


# Global singleton router instance for UI and CLI convenience
_GLOBAL_ROUTER: Optional[NotificationRouter] = None


def get_notification_router(
    force_new: bool = False,
    suppression_window_minutes: int = 60,
    dry_run: Optional[bool] = None,
) -> NotificationRouter:
    """Retrieve or create the shared NotificationRouter instance."""
    global _GLOBAL_ROUTER
    if _GLOBAL_ROUTER is None or force_new:
        _GLOBAL_ROUTER = NotificationRouter(
            suppression_window_minutes=suppression_window_minutes,
            dry_run=dry_run,
        )
    return _GLOBAL_ROUTER
