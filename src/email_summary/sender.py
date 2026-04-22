"""
sender.py
Handles sending emails via SMTP.
"""
import smtplib
import json
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

DEFAULT_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "",
    "sender_password": "",  # Gmail App Password
    "recipients": [],
    "recipient_name": "Manager",
    "risk_profile": "moderate",
    "enabled": False,
}


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            saved = json.load(f)
        # Merge with defaults to handle new keys
        config = {**DEFAULT_CONFIG, **saved}
        return config
    return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def send_email(html_content: str, config: dict = None) -> str:
    """
    Send an HTML email using SMTP.
    Returns "ok" on success, or the error message on failure.
    """
    if config is None:
        config = load_config()

    sender = config.get("sender_email", "")
    password = config.get("sender_password", "")
    recipients = config.get("recipients", [])
    smtp_server = config.get("smtp_server", "smtp.gmail.com")
    smtp_port = config.get("smtp_port", 587)

    if not sender or not password:
        return "Sender email or password not configured."
    if not recipients:
        return "No recipients configured."

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Enkay Investments — Daily Summary"
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)

    msg.attach(MIMEText(html_content, "html"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
        return "ok"
    except Exception as e:
        return str(e)
