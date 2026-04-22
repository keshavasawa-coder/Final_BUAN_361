"""
generator.py
Generates HTML email content from portfolio and client insights data.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

_SRC_DIR = os.path.join(os.path.dirname(__file__), "..")
for _p in [_SRC_DIR, os.path.join(_SRC_DIR, "analysis"), os.path.join(_SRC_DIR, "scoring")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
RANKED_FILE = os.path.join(BASE_DIR, "data", "processed", "ranked_funds.csv")


def _load_ranked():
    return pd.read_csv(RANKED_FILE)


def _get_portfolio_insights(portfolio_df, ranked_df, risk_profile="moderate"):
    """Compute exposure vs quality analysis for portfolio."""
    from analysis.portfolio_review import flag_underperforming_schemes

    result = flag_underperforming_schemes(
        portfolio_df, ranked_df,
        risk_profile=risk_profile,
        aum_threshold=0,
        include_brokerage_flag=True,
    )

    holdings = result.get("all_holdings", pd.DataFrame())
    if holdings.empty or "composite_score" not in holdings.columns:
        return None

    h = holdings.dropna(subset=["composite_score"]).copy()
    if h.empty:
        return None

    total_aum = h["total"].sum()
    if total_aum == 0:
        return None

    h["exposure_pct"] = (h["total"] / total_aum * 100).round(2)
    h["quality_pct"] = h.groupby("sub_category")["composite_score"].rank(pct=True).mul(100).round(1)

    median_exp = h["exposure_pct"].median()
    median_qual = h["quality_pct"].median()

    def categorize(row):
        he = row["exposure_pct"] >= median_exp
        hq = row["quality_pct"] >= median_qual
        if he and not hq:
            return "Overexposed"
        elif not he and hq:
            return "Underutilized"
        elif he and hq:
            return "Well-balanced"
        else:
            return "Dead Weight"

    h["category"] = h.apply(categorize, axis=1)

    overexposed = h[h["category"] == "Overexposed"].sort_values("exposure_pct", ascending=False)
    underutilized = h[h["category"] == "Underutilized"].sort_values("quality_pct", ascending=False)
    dead_weight = h[h["category"] == "Dead Weight"].sort_values("total", ascending=False)

    return {
        "total_aum": total_aum,
        "total_schemes": len(h),
        "overexposed": overexposed,
        "underutilized": underutilized,
        "dead_weight": dead_weight,
        "counts": h["category"].value_counts().to_dict(),
    }


def _get_client_insights(business_df, gaps, metrics):
    """Extract key client insights for email."""
    if business_df is None or not metrics:
        return None

    high_aum_no_sip = gaps.get("high_aum_no_sip", pd.DataFrame())
    reduced_sip = gaps.get("reduced_sip", pd.DataFrame())
    sip_terminated = gaps.get("sip_terminated", pd.DataFrame())
    no_topup = gaps.get("no_topup", pd.DataFrame())

    return {
        "total_clients": metrics.get("total_clients", 0),
        "total_aum": metrics.get("total_aum", 0),
        "total_sip": metrics.get("live_sip_amount", 0),
        "est_revenue": metrics.get("est_annual_revenue", 0),
        "high_aum_no_sip": high_aum_no_sip,
        "reduced_sip": reduced_sip,
        "sip_terminated": sip_terminated,
        "no_topup": no_topup,
        "gap_counts": {
            "high_aum_no_sip": len(high_aum_no_sip),
            "reduced_sip": len(reduced_sip),
            "sip_terminated": len(sip_terminated),
            "no_topup": len(no_topup),
        },
    }


def _fmt_inr(val, unit="Cr"):
    if unit == "Cr":
        return f"{val / 1e7:,.2f} Cr"
    elif unit == "Lakh":
        return f"{val / 1e5:,.1f} Lakh"
    return f"{val:,.0f}"


def _scheme_rows_html(df, cols, max_rows=10):
    """Render a DataFrame subset as an HTML table."""
    if df.empty:
        return "<p style='color:#64748b;'>None found.</p>"
    rows = df.head(max_rows)
    html = '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
    html += "<tr>"
    for c in cols:
        html += f'<th style="text-align:left;padding:6px 8px;border-bottom:2px solid #e2e8f0;color:#475569;">{c}</th>'
    html += "</tr>"
    for _, row in rows.iterrows():
        html += "<tr>"
        for c in cols:
            val = row.get(c, "")
            if isinstance(val, float) and not pd.isna(val):
                val = f"{val:.1f}" if abs(val) > 1 else f"{val:.2f}"
            elif pd.isna(val):
                val = "-"
            html += f'<td style="padding:6px 8px;border-bottom:1px solid #f1f5f9;">{val}</td>'
        html += "</tr>"
    html += "</table>"
    if len(df) > max_rows:
        html += f'<p style="color:#94a3b8;font-size:12px;">... and {len(df) - max_rows} more</p>'
    return html


def generate_email_html(
    portfolio_df=None,
    business_df=None,
    gaps=None,
    metrics=None,
    risk_profile="moderate",
    recipient_name="Manager",
):
    """Generate the full HTML email summary."""
    ranked_df = _load_ranked()
    now = datetime.now().strftime("%d %B %Y, %I:%M %p")

    # Gather insights
    portfolio_insights = None
    if portfolio_df is not None and not portfolio_df.empty:
        portfolio_insights = _get_portfolio_insights(portfolio_df, ranked_df, risk_profile)

    client_insights = None
    if business_df is not None and gaps and metrics:
        client_insights = _get_client_insights(business_df, gaps, metrics)

    # Build HTML
    html = f"""
    <html>
    <head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f8fafc; margin: 0; padding: 20px; }}
        .container {{ max-width: 700px; margin: 0 auto; background: white; border-radius: 12px;
                      box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #3b82f6, #60a5fa); padding: 28px 32px; color: white; }}
        .header h1 {{ margin: 0; font-size: 22px; font-weight: 700; }}
        .header p {{ margin: 6px 0 0; opacity: 0.85; font-size: 14px; }}
        .body {{ padding: 24px 32px; }}
        .section {{ margin-bottom: 28px; }}
        .section-title {{ font-size: 16px; font-weight: 600; color: #1e293b; margin: 0 0 12px;
                          padding-left: 12px; border-left: 4px solid #3b82f6; }}
        .kpi-row {{ display: flex; gap: 12px; margin-bottom: 16px; }}
        .kpi {{ flex: 1; background: #f1f5f9; border-radius: 8px; padding: 12px; text-align: center; }}
        .kpi .value {{ font-size: 20px; font-weight: 700; color: #3b82f6; }}
        .kpi .label {{ font-size: 11px; color: #64748b; margin-top: 4px; }}
        .alert {{ background: #fef2f2; border: 1px solid #fca5a5; border-radius: 8px;
                  padding: 12px 16px; color: #991b1b; margin: 8px 0; font-size: 13px; }}
        .action {{ background: #ecfdf5; border: 1px solid #86efac; border-radius: 8px;
                   padding: 12px 16px; color: #065f46; margin: 8px 0; font-size: 13px; }}
        .footer {{ background: #f8fafc; padding: 16px 32px; text-align: center;
                   color: #94a3b8; font-size: 12px; border-top: 1px solid #e2e8f0; }}
        table {{ width: 100%; }}
    </style>
    </head>
    <body>
    <div class="container">
        <div class="header">
            <h1>Enkay Investments — Daily Summary</h1>
            <p>{now} | {recipient_name}</p>
        </div>
        <div class="body">
    """

    # ── Portfolio Insights ──
    if portfolio_insights:
        pi = portfolio_insights
        counts = pi["counts"]
        html += """
        <div class="section">
            <div class="section-title">Portfolio Exposure Insights</div>
            <div class="kpi-row">
        """
        for cat, color in [("Overexposed", "#ef4444"), ("Underutilized", "#3b82f6"),
                           ("Well-balanced", "#10b981"), ("Dead Weight", "#94a3b8")]:
            cnt = counts.get(cat, 0)
            html += f"""
                <div class="kpi" style="border-top:3px solid {color};">
                    <div class="value" style="color:{color};">{cnt}</div>
                    <div class="label">{cat}</div>
                </div>
            """
        html += "</div>"  # close kpi-row

        # Overexposed
        over = pi["overexposed"]
        if not over.empty:
            over_display = over.copy()
            over_display["AAUM (L)"] = (over_display["total"] / 1e5).round(1)
            over_display["Weight %"] = over_display["exposure_pct"]
            over_display["Quality %"] = over_display["quality_pct"]
            over_display["Brok %"] = over_display["trail_brokerage_incl_gst"]
            html += '<div class="alert">&#9888; <strong>Overexposed schemes</strong> — high portfolio weight but low quality. Consider reducing:</div>'
            html += _scheme_rows_html(over_display, ["scheme", "AAUM (L)", "Weight %", "Quality %", "composite_score", "Brok %"], 7)
            over_aum = over["total"].sum()
            html += f'<div class="action">&#128161; Total overexposed: &#8377;{_fmt_inr(over_aum)} — review and shift to better-rated funds.</div>'

        # Underutilized
        under = pi["underutilized"]
        if not under.empty:
            under_display = under.copy()
            under_display["AAUM (L)"] = (under_display["total"] / 1e5).round(1)
            under_display["Weight %"] = under_display["exposure_pct"]
            under_display["Quality %"] = under_display["quality_pct"]
            under_display["Brok %"] = under_display["trail_brokerage_incl_gst"]
            html += '<div class="action">&#128200; <strong>Underutilized high-quality schemes</strong> — increase allocation:</div>'
            html += _scheme_rows_html(under_display, ["scheme", "AAUM (L)", "Weight %", "Quality %", "composite_score", "Brok %"], 7)

        html += "</div>"  # close section
    else:
        html += """
        <div class="section">
            <div class="section-title">Portfolio Exposure Insights</div>
            <p style="color:#94a3b8;">No portfolio data available. Upload Scheme_wise_AUM to enable this section.</p>
        </div>
        """

    # ── Client Insights ──
    if client_insights:
        ci = client_insights
        gc = ci["gap_counts"]
        total_gaps = sum(gc.values())

        html += """
        <div class="section">
            <div class="section-title">Client Insights</div>
            <div class="kpi-row">
        """
        html += f'<div class="kpi"><div class="value">{ci["total_clients"]:,}</div><div class="label">Total Clients</div></div>'
        html += f'<div class="kpi"><div class="value">&#8377;{_fmt_inr(ci["total_aum"])}</div><div class="label">Total AUM</div></div>'
        html += f'<div class="kpi"><div class="value">&#8377;{_fmt_inr(ci["total_sip"], "Lakh")}</div><div class="label">Live SIP/month</div></div>'
        html += f'<div class="kpi" style="border-top:3px solid #ef4444;"><div class="value" style="color:#ef4444;">{total_gaps}</div><div class="label">Total Gaps</div></div>'
        html += "</div>"

        # High AUM no SIP
        ha = ci["high_aum_no_sip"]
        if not ha.empty:
            ha_display = ha.copy()
            if "Total_MF_AUM" in ha_display.columns:
                ha_display["AUM (L)"] = (pd.to_numeric(ha_display["Total_MF_AUM"], errors="coerce").fillna(0) / 1e5).round(1)
            cols = ["Group", "AUM (L)"] if "AUM (L)" in ha_display.columns else ["Group"]
            ha_aum = pd.to_numeric(ha.get("Total_MF_AUM", 0), errors="coerce").fillna(0).sum()
            html += f'<div class="alert">&#9888; <strong>{len(ha)} clients with high AUM but NO SIP</strong> (&#8377;{_fmt_inr(ha_aum)} at risk)</div>'
            html += _scheme_rows_html(ha_display, cols, 7)
            html += '<div class="action">&#128161; Reach out to start SIPs — these clients have capital but no recurring investment.</div>'

        # Reduced SIP
        rs = ci["reduced_sip"]
        if not rs.empty:
            html += f'<div class="alert">&#9888; <strong>{len(rs)} clients have reduced their SIP</strong> amounts over the last 2 years.</div>'
            html += '<div class="action">&#128161; Contact these clients to understand why and recover SIP flows.</div>'

        # Terminated SIP
        st_df = ci["sip_terminated"]
        if not st_df.empty:
            html += f'<div class="alert">&#9888; <strong>{len(st_df)} clients have terminated SIPs.</strong></div>'
            html += '<div class="action">&#128161; Re-engage these clients — they were investing but stopped. Understand the reason and restart.</div>'

        # No top-up
        nt = ci["no_topup"]
        if not nt.empty:
            html += f'<div class="action">&#128200; <strong>{len(nt)} active SIP clients</strong> have not registered a top-up — opportunity to increase SIP amounts.</div>'

        html += "</div>"  # close section
    else:
        html += """
        <div class="section">
            <div class="section-title">Client Insights</div>
            <p style="color:#94a3b8;">No client data available. Upload Business Insight Report to enable this section.</p>
        </div>
        """

    # ── Footer ──
    html += f"""
        </div>
        <div class="footer">
            Enkay Investments Fund Analytics | Generated on {now}<br>
            This is an automated summary. Open the dashboard for full interactive analysis.
        </div>
    </div>
    </body>
    </html>
    """

    return html
