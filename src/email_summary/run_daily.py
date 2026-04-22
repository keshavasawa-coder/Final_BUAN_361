"""
run_daily.py
Standalone script to generate and send the daily email summary.
Can be called by cron, systemd timer, or any scheduler.

Usage:
    python -m src.email_summary.run_daily

Cron example (every day at 8 AM):
    0 8 * * * cd /path/to/EnkayInvestmentFund-master && venv/bin/python -m src.email_summary.run_daily
"""
import os
import sys

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "src", "analysis"))
sys.path.insert(0, os.path.join(BASE_DIR, "src", "scoring"))

from email_summary.sender import load_config, send_email
from email_summary.generator_ai import generate_email_html


def main():
    config = load_config()

    if not config.get("enabled"):
        print("[SKIP] Email summary is disabled in config.")
        return

    if not config.get("sender_email") or not config.get("recipients"):
        print("[ERROR] Sender email or recipients not configured.")
        return

    risk_profile = config.get("risk_profile", "moderate")
    recipient_name = config.get("recipient_name", "Manager")

    # Load portfolio data (default Scheme_wise_AUM.xls)
    portfolio_df = None
    aum_file = os.path.join(BASE_DIR, "Scheme_wise_AUM.xls")
    if os.path.exists(aum_file):
        from analysis.portfolio_review import load_aum_data
        portfolio_df = load_aum_data()
        print(f"  Portfolio loaded: {len(portfolio_df)} schemes")

    # Client data: try loading from last saved session (if any saved CSVs exist)
    business_df = None
    gaps = None
    metrics = None
    # Client data requires uploaded Excel files — not available in CLI mode.
    # To include client insights, save processed data from the dashboard first.

    print("Generating email summary...")
    html = generate_email_html(
        portfolio_df=portfolio_df,
        business_df=business_df,
        gaps=gaps,
        metrics=metrics,
        risk_profile=risk_profile,
        recipient_name=recipient_name,
    )

    print(f"Sending to: {', '.join(config['recipients'])}")
    result = send_email(html, config)

    if result == "ok":
        print("[OK] Email sent successfully!")
    else:
        print(f"[ERROR] Failed to send: {result}")


if __name__ == "__main__":
    main()
