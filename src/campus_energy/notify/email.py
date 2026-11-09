"""Email alert delivery notifier for Nivara using standard library smtplib and email.mime."""
from __future__ import annotations

import html
import logging
import os
import smtplib
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

from campus_energy.notify.base import BaseNotifier, NotificationMessage, NotificationResult

logger = logging.getLogger(__name__)


class EmailNotifier(BaseNotifier):
    """Dispatches formatted tabular HTML alert notifications via SMTP."""

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_pass: Optional[str] = None,
        alert_email_to: Optional[str] = None,
        from_email: Optional[str] = None,
        dry_run: Optional[bool] = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        """Initialize Email/SMTP notifier.

        Args:
            smtp_host: SMTP server hostname. Defaults to os.environ["SMTP_HOST"].
            smtp_port: SMTP server port (default 587 or os.environ["SMTP_PORT"]).
            smtp_user: SMTP authentication username/email. Defaults to os.environ["SMTP_USER"].
            smtp_pass: SMTP authentication password/token. Defaults to os.environ["SMTP_PASS"].
            alert_email_to: Comma-separated destination email(s). Defaults to os.environ["ALERT_EMAIL_TO"].
            from_email: From address. Defaults to smtp_user or "nivara-alerts@campus.edu".
            dry_run: If True, forces dry-run mode without outbound network connections.
                If None, defaults to dry-run when credentials are missing.
            timeout_seconds: Socket timeout in seconds.
        """
        self._smtp_host = (smtp_host or os.environ.get("SMTP_HOST", "")).strip()

        env_port = os.environ.get("SMTP_PORT", "").strip()
        if smtp_port is not None:
            self._smtp_port = int(smtp_port)
        elif env_port.isdigit():
            self._smtp_port = int(env_port)
        else:
            self._smtp_port = 587

        self._smtp_user = (smtp_user or os.environ.get("SMTP_USER", "")).strip()
        self._smtp_pass = (smtp_pass or os.environ.get("SMTP_PASS", "")).strip()
        self._alert_email_to = (alert_email_to or os.environ.get("ALERT_EMAIL_TO", "")).strip()
        self._from_email = (from_email or self._smtp_user or "nivara-alerts@campus.edu").strip()
        self._timeout_seconds = timeout_seconds

        if dry_run is not None:
            self._dry_run = dry_run
        else:
            self._dry_run = not (
                bool(self._smtp_host)
                and bool(self._smtp_user)
                and bool(self._smtp_pass)
                and bool(self._alert_email_to)
            )

    @property
    def channel_name(self) -> str:
        return "email"

    @property
    def is_live(self) -> bool:
        """True if full SMTP configuration is present and not in dry_run mode."""
        return bool(
            self._smtp_host
            and self._smtp_user
            and self._smtp_pass
            and self._alert_email_to
            and not self._dry_run
        )

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    def get_recipient_list(self) -> List[str]:
        """Parse destination email addresses into a clean list."""
        if not self._alert_email_to:
            return ["facility-operations@campus.edu"]
        return [email.strip() for email in self._alert_email_to.split(",") if email.strip()]

    def format_plain_text(self, message: NotificationMessage) -> str:
        """Format plain text fallback representation."""
        lines = [
            f"NIVARA CAMPUS ENERGY ALERT — [{message.severity.upper()}]",
            f"Title: {message.title}",
            f"Asset ID: {message.asset_id}",
            f"Recipient Role: {message.recipient_role.upper()}",
            f"Timestamp: {message.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
            f"Work Order: {message.work_order_id or 'None'}",
            "",
            "DETAILS:",
            message.description,
            "",
            f"ESTIMATED IMPACT / SAVINGS: {message.impact_range}",
            f"SERVICE GUARD ASSESSMENT: {message.service_guard_note}",
            "",
            "--",
            "Nivara Campus Energy Intelligence Platform",
        ]
        return "\n".join(lines)

    def format_html(self, message: NotificationMessage) -> str:
        """Format tabular, professional HTML email message."""
        sev_colors = {
            "critical": ("#dc2626", "#fef2f2", "#991b1b"),
            "high": ("#ea580c", "#fff7ed", "#9a3412"),
            "medium": ("#d97706", "#fffbeb", "#92400e"),
            "low": ("#2563eb", "#eff6ff", "#1e40af"),
        }
        accent_color, bg_light, text_dark = sev_colors.get(
            message.severity.lower(), ("#2563eb", "#eff6ff", "#1e40af")
        )

        title_safe = html.escape(message.title)
        asset_safe = html.escape(message.asset_id)
        desc_safe = html.escape(message.description).replace("\n", "<br/>")
        impact_safe = html.escape(message.impact_range)
        guard_safe = html.escape(message.service_guard_note)
        role_safe = html.escape(message.recipient_role.upper())
        time_str = message.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        wo_display = html.escape(message.work_order_id) if message.work_order_id else "<em>None (Advisory)</em>"

        return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Nivara Energy Alert</title>
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f1f5f9; margin: 0; padding: 24px; color: #1e293b;">
  <div style="max-width: 640px; margin: 0 auto; background: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border: 1px solid #e2e8f0;">
    
    <!-- Header Banner -->
    <div style="background-color: {accent_color}; color: #ffffff; padding: 18px 24px;">
      <div style="font-size: 11px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; opacity: 0.9;">
        Nivara Campus Energy Intelligence • Outbound Notification
      </div>
      <h2 style="margin: 6px 0 0 0; font-size: 20px; font-weight: 700;">
        [{message.severity.upper()}] {title_safe}
      </h2>
    </div>

    <!-- Main Content -->
    <div style="padding: 24px;">
      <p style="font-size: 15px; line-height: 1.5; color: #334155; margin-top: 0; margin-bottom: 20px;">
        {desc_safe}
      </p>

      <!-- Tabular Summary -->
      <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 14px;">
        <tr style="border-bottom: 1px solid #e2e8f0;">
          <td style="padding: 10px 12px; font-weight: 600; color: #64748b; width: 38%;">Target Asset</td>
          <td style="padding: 10px 12px; font-family: monospace; font-weight: 600; color: #0f172a;">{asset_safe}</td>
        </tr>
        <tr style="border-bottom: 1px solid #e2e8f0; background-color: #f8fafc;">
          <td style="padding: 10px 12px; font-weight: 600; color: #64748b;">Recipient Role</td>
          <td style="padding: 10px 12px; font-weight: 600; color: {text_dark};">{role_safe}</td>
        </tr>
        <tr style="border-bottom: 1px solid #e2e8f0;">
          <td style="padding: 10px 12px; font-weight: 600; color: #64748b;">Detection Time</td>
          <td style="padding: 10px 12px; color: #334155;">{time_str}</td>
        </tr>
        <tr style="border-bottom: 1px solid #e2e8f0; background-color: #f8fafc;">
          <td style="padding: 10px 12px; font-weight: 600; color: #64748b;">Work Order Reference</td>
          <td style="padding: 10px 12px; font-family: monospace; color: #0f172a;">{wo_display}</td>
        </tr>
        <tr style="border-bottom: 1px solid #e2e8f0;">
          <td style="padding: 10px 12px; font-weight: 600; color: #64748b;">Estimated Impact / Savings</td>
          <td style="padding: 10px 12px; font-weight: 700; color: #0f172a;">{impact_safe}</td>
        </tr>
        <tr style="background-color: {bg_light};">
          <td style="padding: 12px; font-weight: 600; color: {text_dark};">Service Guard Status</td>
          <td style="padding: 12px; font-weight: 600; color: {text_dark};">{guard_safe}</td>
        </tr>
      </table>

      <!-- Footer Note -->
      <div style="font-size: 12px; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 16px;">
        This automated operational dispatch was generated by Nivara based on facility submeter telemetry,
        IPMVP Option B counterfactual models, and Service-Impact Guard constraints.
      </div>
    </div>
  </div>
</body>
</html>
"""

    def send(self, message: NotificationMessage) -> NotificationResult:
        """Dispatch message via SMTP or simulate in dry-run mode."""
        recipients = self.get_recipient_list()
        subject = f"[Nivara Alert] [{message.severity.upper()}] {message.title} ({message.asset_id})"

        if not self.is_live:
            simulated_id = f"DRYRUN-EMAIL-{uuid.uuid4().hex[:8].upper()}"
            logger.info(
                "[EmailNotifier DRY-RUN] Alert to %s (Subject: %s | ID: %s)",
                ", ".join(recipients),
                subject,
                simulated_id,
            )
            return NotificationResult(
                success=True,
                channel=self.channel_name,
                dry_run=True,
                message_id=simulated_id,
                error=None,
            )

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self._from_email
        msg["To"] = ", ".join(recipients)

        part_plain = MIMEText(self.format_plain_text(message), "plain", "utf-8")
        part_html = MIMEText(self.format_html(message), "html", "utf-8")
        msg.attach(part_plain)
        msg.attach(part_html)

        try:
            if self._smtp_port == 465:
                smtp_server = smtplib.SMTP_SSL(self._smtp_host, self._smtp_port, timeout=self._timeout_seconds)
            else:
                smtp_server = smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=self._timeout_seconds)
                smtp_server.ehlo()
                smtp_server.starttls()
                smtp_server.ehlo()

            with smtp_server:
                if self._smtp_user and self._smtp_pass:
                    smtp_server.login(self._smtp_user, self._smtp_pass)
                smtp_server.send_message(msg)

            msg_id = f"SMTP-{uuid.uuid4().hex[:8].upper()}"
            logger.info("[EmailNotifier] Successfully sent email to %s (ID: %s)", recipients, msg_id)
            return NotificationResult(
                success=True,
                channel=self.channel_name,
                dry_run=False,
                message_id=msg_id,
                error=None,
            )
        except Exception as e:
            logger.exception("[EmailNotifier] SMTP delivery failed: %s", e)
            return NotificationResult(
                success=False,
                channel=self.channel_name,
                dry_run=False,
                message_id=None,
                error=str(e),
            )
