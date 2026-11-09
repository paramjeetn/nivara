"""Telegram alert delivery notifier for Nivara using urllib.request."""
from __future__ import annotations

import html
import json
import logging
import os
import urllib.error
import urllib.request
import uuid
from typing import Optional

from campus_energy.notify.base import BaseNotifier, NotificationMessage, NotificationResult

logger = logging.getLogger(__name__)


class TelegramNotifier(BaseNotifier):
    """Dispatches real-time alerts to facility operations via Telegram Bot API."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        dry_run: Optional[bool] = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        """Initialize Telegram notifier.

        Args:
            bot_token: Telegram bot token. Defaults to os.environ["TELEGRAM_BOT_TOKEN"].
            chat_id: Telegram target chat/channel ID. Defaults to os.environ["TELEGRAM_CHAT_ID"].
            dry_run: If True, forces dry-run mode without outbound HTTP calls.
                If None, defaults to dry-run when credentials are missing.
            timeout_seconds: Outbound HTTP timeout in seconds.
        """
        self._bot_token = (bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
        self._chat_id = (chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")).strip()
        self._timeout_seconds = timeout_seconds

        if dry_run is not None:
            self._dry_run = dry_run
        else:
            # If credentials are not present, fallback automatically to dry_run
            self._dry_run = not (bool(self._bot_token) and bool(self._chat_id))

    @property
    def channel_name(self) -> str:
        return "telegram"

    @property
    def is_live(self) -> bool:
        """True if real credentials are present and not in dry_run mode."""
        return bool(self._bot_token and self._chat_id and not self._dry_run)

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    def format_message(self, message: NotificationMessage) -> str:
        """Format a professional, high-visibility HTML alert for Telegram."""
        sev_icons = {
            "critical": "🚨",
            "high": "⚠️",
            "medium": "🟡",
            "low": "ℹ️",
        }
        icon = sev_icons.get(message.severity.lower(), "ℹ️")

        title_safe = html.escape(message.title)
        asset_safe = html.escape(message.asset_id)
        desc_safe = html.escape(message.description)
        impact_safe = html.escape(message.impact_range)
        guard_safe = html.escape(message.service_guard_note)
        role_safe = html.escape(message.recipient_role.upper())
        time_str = message.timestamp.strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            f"{icon} <b>[NIVARA ALERT] {title_safe}</b>",
            f"<b>Severity:</b> <code>{message.severity.upper()}</code> | <b>Role:</b> <code>{role_safe}</code>",
            f"<b>Asset:</b> <code>{asset_safe}</code>",
            f"<b>Detected:</b> <code>{time_str}</code>",
            "",
            "📝 <b>Actionable Details:</b>",
            f"{desc_safe}",
            "",
            "💰 <b>Estimated Impact / Savings:</b>",
            f"<code>{impact_safe}</code>",
            "",
            "🛡️ <b>Service Guard Status:</b>",
            f"{guard_safe}",
        ]

        if message.work_order_id:
            wo_safe = html.escape(message.work_order_id)
            lines.append("")
            lines.append(f"📋 <b>Work Order Reference:</b> <code>{wo_safe}</code>")

        lines.append("")
        lines.append("<i>Nivara Campus Energy Intelligence Platform</i>")
        return "\n".join(lines)

    def send(self, message: NotificationMessage) -> NotificationResult:
        """Dispatch message via Telegram Bot API or simulate in dry-run mode."""
        formatted_text = self.format_message(message)

        if not self.is_live:
            simulated_id = f"DRYRUN-TG-{uuid.uuid4().hex[:8].upper()}"
            target_chat = self._chat_id or "[ENV: TELEGRAM_CHAT_ID (Unset)]"
            logger.info(
                "[TelegramNotifier DRY-RUN] Message to %s (ID: %s):\n%s",
                target_chat,
                simulated_id,
                formatted_text,
            )
            return NotificationResult(
                success=True,
                channel=self.channel_name,
                dry_run=True,
                message_id=simulated_id,
                error=None,
            )

        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": formatted_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout_seconds) as response:
                resp_bytes = response.read()
                resp_json = json.loads(resp_bytes.decode("utf-8"))

            if resp_json.get("ok"):
                msg_id = str(resp_json.get("result", {}).get("message_id", "LIVE-TG-SENT"))
                logger.info("[TelegramNotifier] Successfully sent message %s to chat %s", msg_id, self._chat_id)
                return NotificationResult(
                    success=True,
                    channel=self.channel_name,
                    dry_run=False,
                    message_id=msg_id,
                    error=None,
                )
            else:
                err_desc = resp_json.get("description", "Unknown Telegram API rejection")
                logger.error("[TelegramNotifier] Telegram API returned error: %s", err_desc)
                return NotificationResult(
                    success=False,
                    channel=self.channel_name,
                    dry_run=False,
                    message_id=None,
                    error=err_desc,
                )
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            err_msg = f"HTTP {e.code} {e.reason}: {err_body}"
            logger.error("[TelegramNotifier] HTTP failure: %s", err_msg)
            return NotificationResult(
                success=False,
                channel=self.channel_name,
                dry_run=False,
                message_id=None,
                error=err_msg,
            )
        except Exception as e:
            logger.exception("[TelegramNotifier] Dispatch failed: %s", e)
            return NotificationResult(
                success=False,
                channel=self.channel_name,
                dry_run=False,
                message_id=None,
                error=str(e),
            )
