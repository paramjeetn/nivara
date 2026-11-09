"""Comprehensive unit tests for Nivara Notification Subsystem (Step 11)."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from campus_energy.decision.ledger import WorkOrderItem
from campus_energy.decision.recommendation import (
    GuardResult,
    Recommendation,
    SavingsBundle,
    SavingsRange,
)
from campus_energy.notify.base import (
    BaseNotifier,
    NotificationMessage,
    NotificationResult,
)
from campus_energy.notify.email import EmailNotifier
from campus_energy.notify.router import NotificationRouter
from campus_energy.notify.telegram import TelegramNotifier
from campus_energy.schemas import Alert, AlertStatus, Severity


@pytest.fixture
def sample_message() -> NotificationMessage:
    """Fixture creating a standard NotificationMessage."""
    return NotificationMessage(
        recipient_role="operations",
        severity="critical",
        title="High Chiller Baseline Power Surge",
        asset_id="CHILLER-01",
        description="Chiller baseline consumption exceeded expected 45 kW by +32 kW during unoccupied hours.",
        impact_range="₹2,400 [₹1,920 – ₹2,880] / day",
        service_guard_note="🟢 Service-Safe: Non-critical off-hours load",
        work_order_id="WO-2026-001",
        timestamp=datetime(2026, 11, 9, 10, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_work_order() -> WorkOrderItem:
    """Fixture creating an actionable WorkOrderItem."""
    savings = SavingsBundle.from_expected(
        kwh_expected=120.0,
        inr_per_kwh=8.0,
        co2_per_kwh=0.82,
        spread=0.20,
    )
    guard = GuardResult(
        status="pass",
        reason="Scheduled setback during unoccupied window; protected clinic unaffected.",
        constraints_checked=["protected_load_check", "pmv_comfort_check"],
    )
    rec = Recommendation(
        id="REC-2026-042",
        facility_id="university",
        type="anomaly_fix",
        target="AHU-NORTH-03",
        action_text="Reset supply fan static pressure setpoint from 2.5 in.wg to 1.4 in.wg.",
        trigger="high_consumption anomaly detected on AHU-NORTH-03",
        savings=savings,
        cost=0.0,
        payback_months=0.0,
        service_guard=guard,
        owner_role="operations",
        lifecycle="work_order",
    )
    return WorkOrderItem(
        work_order_id="WO-2026-042",
        recommendation_id="REC-2026-042",
        recommendation=rec,
        status="work_order",
        priority="high",
        assignee="HVAC Controls Crew",
        created_at=datetime(2026, 11, 9, 8, 30, tzinfo=timezone.utc),
        sla_deadline=datetime(2026, 11, 9, 20, 30, tzinfo=timezone.utc),
        estimated_savings=savings,
        mv_method="IPMVP Option B / Counterfactual Baseline",
    )


# =============================================================================
# 1. Base Models and Interface Tests
# =============================================================================

def test_notification_message_serialization(sample_message: NotificationMessage):
    """Verify NotificationMessage fields and serialization integrity."""
    dumped = sample_message.model_dump()
    assert dumped["asset_id"] == "CHILLER-01"
    assert dumped["severity"] == "critical"
    assert dumped["recipient_role"] == "operations"
    assert "₹2,400" in dumped["impact_range"]
    assert "Service-Safe" in dumped["service_guard_note"]
    assert dumped["work_order_id"] == "WO-2026-001"

    # Round-trip JSON validation
    json_str = sample_message.model_dump_json()
    reloaded = NotificationMessage.model_validate_json(json_str)
    assert reloaded.title == sample_message.title
    assert reloaded.timestamp == sample_message.timestamp


def test_custom_notifier_implementation():
    """Verify that custom subclass correctly fulfills BaseNotifier contract."""
    class MockNotifier(BaseNotifier):
        @property
        def channel_name(self) -> str:
            return "mock_channel"

        @property
        def is_live(self) -> bool:
            return False

        def send(self, message: NotificationMessage) -> NotificationResult:
            return NotificationResult(
                success=True,
                channel=self.channel_name,
                dry_run=True,
                message_id="MOCK-001",
            )

    notifier = MockNotifier()
    assert notifier.channel_name == "mock_channel"
    assert not notifier.is_live
    res = notifier.send(NotificationMessage(
        title="Test",
        asset_id="A-1",
        description="Desc",
        impact_range="0",
        service_guard_note="OK",
    ))
    assert res.success is True
    assert res.channel == "mock_channel"
    assert res.dry_run is True


# =============================================================================
# 2. Telegram Notifier Tests
# =============================================================================

def test_telegram_dry_run_without_credentials(sample_message: NotificationMessage):
    """Verify TelegramNotifier operates safely in dry-run mode when env vars are absent."""
    with patch.dict(os.environ, {}, clear=True):
        tg = TelegramNotifier()
        assert not tg.is_live
        assert tg.dry_run is True
        assert tg.channel_name == "telegram"

        with patch("urllib.request.urlopen") as mock_urlopen:
            res = tg.send(sample_message)
            mock_urlopen.assert_not_called()
            assert res.success is True
            assert res.dry_run is True
            assert res.channel == "telegram"
            assert res.message_id is not None
            assert res.message_id.startswith("DRYRUN-TG-")
            assert res.error is None


def test_telegram_explicit_dry_run(sample_message: NotificationMessage):
    """Verify dry_run=True overrides present credentials."""
    tg = TelegramNotifier(bot_token="fake_token", chat_id="12345", dry_run=True)
    assert not tg.is_live
    res = tg.send(sample_message)
    assert res.success is True
    assert res.dry_run is True


def test_telegram_formatting_contains_required_fields(sample_message: NotificationMessage):
    """Verify Telegram HTML alert contains all relevant operational fields and emojis."""
    tg = TelegramNotifier()
    formatted = tg.format_message(sample_message)

    assert "🚨" in formatted
    assert "CRITICAL" in formatted
    assert "CHILLER-01" in formatted
    assert "₹2,400 [₹1,920 – ₹2,880] / day" in formatted
    assert "🟢 Service-Safe" in formatted
    assert "WO-2026-001" in formatted
    assert "High Chiller Baseline Power Surge" in formatted


def test_telegram_live_dispatch_success(sample_message: NotificationMessage):
    """Verify live HTTP request dispatch when credentials are set."""
    tg = TelegramNotifier(bot_token="valid_token", chat_id="chat_999", dry_run=False)
    assert tg.is_live

    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"ok": true, "result": {"message_id": 98765}}'
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        res = tg.send(sample_message)
        assert mock_urlopen.called
        assert res.success is True
        assert res.dry_run is False
        assert res.message_id == "98765"
        assert res.error is None


def test_telegram_live_dispatch_api_error(sample_message: NotificationMessage):
    """Verify graceful handling when Telegram Bot API rejects request."""
    tg = TelegramNotifier(bot_token="bad_token", chat_id="chat_999", dry_run=False)
    assert tg.is_live

    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"ok": false, "description": "Unauthorized"}'
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = tg.send(sample_message)
        assert res.success is False
        assert res.dry_run is False
        assert "Unauthorized" in (res.error or "")


# =============================================================================
# 3. Email Notifier Tests
# =============================================================================

def test_email_dry_run_without_credentials(sample_message: NotificationMessage):
    """Verify EmailNotifier defaults to dry-run mode and performs no SMTP network calls."""
    with patch.dict(os.environ, {}, clear=True):
        em = EmailNotifier()
        assert not em.is_live
        assert em.dry_run is True
        assert em.channel_name == "email"

        with patch("smtplib.SMTP") as mock_smtp:
            res = em.send(sample_message)
            mock_smtp.assert_not_called()
            assert res.success is True
            assert res.dry_run is True
            assert res.channel == "email"
            assert res.message_id is not None
            assert res.message_id.startswith("DRYRUN-EMAIL-")


def test_email_html_and_text_formatting(sample_message: NotificationMessage):
    """Verify email HTML body includes table structure and Service Guard status."""
    em = EmailNotifier()
    html_body = em.format_html(sample_message)
    text_body = em.format_plain_text(sample_message)

    # HTML assertions
    assert "<table" in html_body
    assert "CHILLER-01" in html_body
    assert "CRITICAL" in html_body
    assert "₹2,400 [₹1,920 – ₹2,880] / day" in html_body
    assert "🟢 Service-Safe" in html_body
    assert "WO-2026-001" in html_body

    # Plain text assertions
    assert "CHILLER-01" in text_body
    assert "OPERATIONS" in text_body


def test_email_live_dispatch_success(sample_message: NotificationMessage):
    """Verify live SMTP dispatch connects, logs in, and delivers email."""
    em = EmailNotifier(
        smtp_host="smtp.campus.edu",
        smtp_port=587,
        smtp_user="alerts@campus.edu",
        smtp_pass="secret123",
        alert_email_to="facility-lead@campus.edu",
        dry_run=False,
    )
    assert em.is_live

    mock_smtp_inst = MagicMock()
    mock_smtp_inst.__enter__.return_value = mock_smtp_inst

    with patch("smtplib.SMTP", return_value=mock_smtp_inst) as mock_smtp_cls:
        res = em.send(sample_message)
        assert mock_smtp_cls.called
        assert mock_smtp_inst.starttls.called
        assert mock_smtp_inst.login.called
        assert mock_smtp_inst.send_message.called
        assert res.success is True
        assert res.dry_run is False
        assert res.message_id is not None


def test_email_live_dispatch_connection_failure(sample_message: NotificationMessage):
    """Verify failure handling when SMTP connection raises an exception."""
    em = EmailNotifier(
        smtp_host="bad.smtp.server",
        smtp_port=587,
        smtp_user="alerts@campus.edu",
        smtp_pass="secret123",
        alert_email_to="ops@campus.edu",
        dry_run=False,
    )
    with patch("smtplib.SMTP", side_effect=ConnectionRefusedError("Connection refused")):
        res = em.send(sample_message)
        assert res.success is False
        assert "Connection refused" in (res.error or "")


# =============================================================================
# 4. Notification Router & Routing Matrix Tests
# =============================================================================

def test_router_status_and_defaults():
    """Verify NotificationRouter initialization and status badge reporting."""
    router = NotificationRouter(dry_run=True, suppression_window_minutes=45)
    status = router.get_gateway_status()

    assert not status["is_live"]
    assert "Dry-Run" in status["status_badge"]
    assert "telegram" in status["registered_channels"]
    assert "email" in status["registered_channels"]
    assert status["suppression_window_minutes"] == 45


def test_router_routing_critical_severity(sample_message: NotificationMessage):
    """Verify critical severity routes to both Telegram and Email."""
    router = NotificationRouter(dry_run=True)
    channels = router.get_channels_for_message(sample_message)
    channel_names = [c.channel_name for c in channels]

    assert "telegram" in channel_names
    assert "email" in channel_names


def test_router_routing_sustainability_role():
    """Verify sustainability role routes to Email."""
    msg = NotificationMessage(
        recipient_role="sustainability",
        severity="medium",
        title="Monthly Carbon Abatement Progress",
        asset_id="CAMPUS-GRID",
        description="Renewable fraction reached 42% this week.",
        impact_range="1,400 kg CO2e / day",
        service_guard_note="🟢 Compliance-Safe: NAPCC target met",
    )
    router = NotificationRouter(dry_run=True)
    channels = router.get_channels_for_message(msg)
    channel_names = [c.channel_name for c in channels]

    assert channel_names == ["email"]


def test_router_routing_administrator_role():
    """Verify administrator role routes to Email for medium, and both for high/critical."""
    router = NotificationRouter(dry_run=True)

    msg_med = NotificationMessage(
        recipient_role="administrator",
        severity="medium",
        title="Department Billing Variance",
        asset_id="BLDG-HOSTEL-01",
        description="Electricity bill exceeded nominal budget by 8%.",
        impact_range="₹12,000 / month",
        service_guard_note="🟢 Safe",
    )
    ch_med = [c.channel_name for c in router.get_channels_for_message(msg_med)]
    assert ch_med == ["email"]

    msg_high = NotificationMessage(
        recipient_role="administrator",
        severity="high",
        title="Peak Demand Surcharge Approaching Threshold",
        asset_id="TRANSFORMER-01",
        description="Peak kVA demand reached 92% of contract demand ceiling.",
        impact_range="₹85,000 potential surcharge",
        service_guard_note="🟢 Safe",
    )
    ch_high = [c.channel_name for c in router.get_channels_for_message(msg_high)]
    assert "telegram" in ch_high
    assert "email" in ch_high


# =============================================================================
# 5. De-duplication & Rate Limiting Tests
# =============================================================================

def test_deduplication_throttles_rapid_duplicates(sample_message: NotificationMessage):
    """Verify sending identical alert within suppression window is throttled."""
    router = NotificationRouter(dry_run=True, suppression_window_minutes=60)
    router.clear_cache()

    # 1st dispatch -> succeeds across channels
    results_1 = router.dispatch(sample_message)
    assert len(results_1) == 2
    assert all(r.success for r in results_1)

    # 2nd dispatch (immediate duplicate) -> throttled
    results_2 = router.dispatch(sample_message)
    assert len(results_2) == 1
    throttled = results_2[0]
    assert throttled.success is False
    assert throttled.channel == "router"
    assert "Throttled" in (throttled.error or "")

    # History should contain both attempts
    hist = router.get_history(limit=10)
    assert len(hist) == 2
    assert hist[0]["success"] is False  # Most recent is the throttled one


def test_deduplication_expires_after_suppression_window(sample_message: NotificationMessage):
    """Verify dispatch succeeds once suppression window has elapsed."""
    router = NotificationRouter(dry_run=True, suppression_window_minutes=30)
    router.clear_cache()

    # Initial dispatch
    results_1 = router.dispatch(sample_message)
    assert all(r.success for r in results_1)

    # Fast forward time beyond 30 min window
    now_future = datetime.now(timezone.utc) + timedelta(minutes=31)
    assert not router.is_throttled(sample_message, now=now_future)

    # Manually back-date cache entry to test actual dispatch
    key = router._get_dedup_key(sample_message)
    router._cache[key] = datetime.now(timezone.utc) - timedelta(minutes=35)

    results_3 = router.dispatch(sample_message)
    assert len(results_3) == 2
    assert all(r.success for r in results_3)


# =============================================================================
# 6. Dispatch from WorkOrderItem and Alert Tests
# =============================================================================

def test_dispatch_from_work_order(sample_work_order: WorkOrderItem):
    """Verify dispatch_from_work_order generates correct message format and guard badges."""
    router = NotificationRouter(dry_run=True)
    router.clear_cache()

    results = router.dispatch_from_work_order(sample_work_order)
    assert len(results) == 2
    assert all(r.success for r in results)

    # Inspect history to check constructed payload
    hist = router.get_history(limit=1)
    record = hist[0]

    assert record["asset_id"] == "AHU-NORTH-03"
    assert record["work_order_id"] == "WO-2026-042"
    assert record["severity"] == "high"
    assert "₹" in record["impact_range"]
    assert "🟢 Service-Safe" in record["service_guard_note"]
    assert "static pressure setpoint" in record["title"]


def test_dispatch_from_alert_object():
    """Verify dispatch_from_alert builds valid notification from an Alert schema."""
    alert = Alert(
        alert_id="ALT-2026-TEST",
        detected_at=datetime.now(timezone.utc),
        asset_id="CHILLER-01",
        building_id="BLDG-ENGINEERING",
        anomaly_type="after_hours_consumption",
        severity=Severity.HIGH,
        deviation=28.5,
        evidence="Active baseline 34 kW detected at 02:30 AM on Sunday.",
        recommendation="Check BAS setback schedule for Chiller #1.",
        status=AlertStatus.ACTIVE,
    )

    router = NotificationRouter(dry_run=True)
    router.clear_cache()

    results = router.dispatch_from_alert(alert)
    assert len(results) == 2
    assert all(r.success for r in results)

    hist = router.get_history(limit=1)
    record = hist[0]
    assert record["asset_id"] == "CHILLER-01"
    assert "After Hours Consumption" in record["title"]
    assert record["severity"] == "high"
    assert "Dev: 28.5 kWh" in record["impact_range"]
