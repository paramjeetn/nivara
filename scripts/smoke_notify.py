#!/usr/bin/env python3
"""Smoke test utility for Nivara notification gateway (Telegram & Email/SMTP).

Inspects environment variables:
  - Telegram: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  - Email: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_EMAIL_TO

If credentials are set, performs real outbound network dispatches;
if unset, safely runs a realistic dry-run dispatch and prints formatted previews.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from campus_energy.notify import (
    EmailNotifier,
    NotificationMessage,
    NotificationRouter,
    TelegramNotifier,
)


def main() -> int:
    print("=" * 70)
    print("  NIVARA NOTIFICATION GATEWAY — SMOKE TEST & VERIFICATION")
    print("=" * 70)

    # 1. Inspect Environment Variables
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    smtp_host = os.environ.get("SMTP_HOST", "").strip()
    smtp_user = os.environ.get("SMTP_USER", "").strip()
    smtp_pass = os.environ.get("SMTP_PASS", "").strip()
    alert_to = os.environ.get("ALERT_EMAIL_TO", "").strip()

    print("\n🔍 Environment Inspection:")
    print(f"  • TELEGRAM_BOT_TOKEN: {'✅ Set' if tg_token else '❌ Unset (Dry-run mode)'}")
    print(f"  • TELEGRAM_CHAT_ID:   {'✅ Set' if tg_chat else '❌ Unset (Dry-run mode)'}")
    print(f"  • SMTP_HOST:          {'✅ Set' if smtp_host else '❌ Unset (Dry-run mode)'}")
    print(f"  • SMTP_USER:          {'✅ Set' if smtp_user else '❌ Unset (Dry-run mode)'}")
    print(f"  • SMTP_PASS:          {'✅ Set' if smtp_pass else '❌ Unset (Dry-run mode)'}")
    print(f"  • ALERT_EMAIL_TO:     {'✅ Set' if alert_to else '❌ Unset (Dry-run mode)'}")

    # 2. Instantiate Notifiers & Router
    tg_notifier = TelegramNotifier()
    em_notifier = EmailNotifier()
    router = NotificationRouter(notifiers=[tg_notifier, em_notifier], suppression_window_minutes=60)

    gateway_status = router.get_gateway_status()
    print(f"\n📡 Gateway Status: {gateway_status['status_badge']}")
    print(f"  • Telegram Channel: {'🟢 LIVE' if tg_notifier.is_live else '🟡 SIMULATION (Dry-Run)'}")
    print(f"  • Email Channel:    {'🟢 LIVE' if em_notifier.is_live else '🟡 SIMULATION (Dry-Run)'}")

    # 3. Formulate Realistic Test Notification
    test_msg = NotificationMessage(
        recipient_role="operations",
        severity="critical",
        title="Chiller Plant #2 Low Delta-T Anomaly & Chilled Water Loop Bypass",
        asset_id="CHILLER-PLANT-02",
        description=(
            "Primary chilled water loop delta-T collapsed to 1.8°C (expected >= 5.5°C). "
            "Telemetry indicates three-way bypass valve stuck open at 85% during peak campus cooling."
        ),
        impact_range="₹3,450 [₹2,760 – ₹4,140] / day",
        service_guard_note="🟢 Service-Safe: Non-critical off-hours HVAC load with redundant backup chiller online",
        work_order_id="WO-2026-SMOKE-01",
        timestamp=datetime.now(timezone.utc),
    )

    print("\n📦 Test Notification Payload:")
    print(f"  • Title:        {test_msg.title}")
    print(f"  • Asset:        {test_msg.asset_id}")
    print(f"  • Severity:     {test_msg.severity.upper()}")
    print(f"  • Role:         {test_msg.recipient_role.upper()}")
    print(f"  • Impact Range: {test_msg.impact_range}")
    print(f"  • Service Guard:{test_msg.service_guard_note}")
    print(f"  • Work Order:   {test_msg.work_order_id}")

    # 4. Preview Message Formatters
    print("\n💬 Telegram Message Preview (HTML):")
    print("-" * 50)
    print(tg_notifier.format_message(test_msg))
    print("-" * 50)

    # 5. Dispatch
    print("\n🚀 Dispatched through NotificationRouter...")
    results = router.dispatch(test_msg)

    print("\n📋 Dispatch Results:")
    all_succeeded = True
    for res in results:
        status_icon = "✅" if res.success else "❌"
        mode_label = "DRY-RUN" if res.dry_run else "LIVE NETWORK"
        print(f"  {status_icon} [{res.channel.upper()}] Mode: {mode_label} | Success: {res.success}")
        if res.message_id:
            print(f"      Message ID: {res.message_id}")
        if res.error:
            print(f"      Error: {res.error}")
            all_succeeded = False

    print("\n" + "=" * 70)
    if all_succeeded:
        print("  🎉 SMOKE TEST PASSED: All notification channels dispatched successfully.")
        print("=" * 70)
        return 0
    else:
        print("  ⚠️ SMOKE TEST COMPLETED WITH FAILURES.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
