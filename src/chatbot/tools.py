import os
import sys
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from agents import function_tool

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

# Ensure analysis & scoring modules are importable
_SRC_DIR = os.path.join(os.path.dirname(__file__), "..")
for _p in [_SRC_DIR, os.path.join(_SRC_DIR, "analysis"), os.path.join(_SRC_DIR, "scoring")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ranked_df = None
_master_df = None

# ── Client data holder (populated from Streamlit session_state) ─────────────
# Call set_client_data() from the Streamlit app whenever client data is available.
_client_data = {
    "business_df": None,   # ci_business_df from session_state
    "sip_df": None,        # ci_sip_df from session_state
    "gaps": None,           # ci_gaps from session_state
    "metrics": None,        # ci_metrics from session_state
    "pareto": None,         # ci_pareto from session_state
}


def set_client_data(business_df=None, sip_df=None, gaps=None, metrics=None, pareto=None):
    """Called from the Streamlit app to inject uploaded client data into the tools."""
    _client_data["business_df"] = business_df
    _client_data["sip_df"] = sip_df
    _client_data["gaps"] = gaps
    _client_data["metrics"] = metrics
    _client_data["pareto"] = pareto


def _load_ranked() -> pd.DataFrame:
    global _ranked_df
    if _ranked_df is None:
        path = os.path.join(DATA_DIR, "ranked_funds.csv")
        _ranked_df = pd.read_csv(path)
    return _ranked_df


def _load_master() -> pd.DataFrame:
    global _master_df
    if _master_df is None:
        path = os.path.join(DATA_DIR, "master_scheme_table.csv")
        _master_df = pd.read_csv(path)
    return _master_df


METRIC_COLUMNS = {
    "brokerage": "trail_brokerage_incl_gst",
    "score": "composite_score",
    "aum": "aum_cr",
    "return_1y": "return_1y_regular",
    "return_3y": "return_3y_regular",
    "return_5y": "return_5y_regular",
    "rank": "rank",
}

DISPLAY_COLS = [
    "scheme_name", "amc", "category", "sub_category",
    "composite_score", "rank", "risk_profile",
    "return_1y_regular", "return_3y_regular", "return_5y_regular",
    "trail_brokerage_incl_gst", "aum_cr", "tieup_category",
    "score_return", "score_brokerage", "score_aum", "score_tieup",
]


@function_tool
def get_top_funds(
    metric: str = "score",
    top_n: int = 10,
    category: str = "",
    sub_category: str = "",
    risk_profile: str = "moderate",
    ascending: bool = False,
) -> str:
    """Get the top N funds ranked by a given metric.

    Args:
        metric: Metric to rank by. One of: brokerage, score, aum, return_1y, return_3y, return_5y, rank.
        top_n: Number of funds to return (default 10, max 25).
        category: Filter by category (Equity, Debt, Hybrid, etc). Empty string means all.
        sub_category: Filter by sub-category (Large Cap Fund, Liquid Fund, etc). Empty string means all.
        risk_profile: Risk profile for scoring: conservative, moderate, or aggressive.
        ascending: If True, sort ascending (lowest first). Default False (highest first).
    """
    df = _load_ranked()
    top_n = min(top_n, 25)

    col = METRIC_COLUMNS.get(metric, "composite_score")
    df = df[df["risk_profile"] == risk_profile].copy()

    if category:
        df = df[df["category"].str.lower() == category.lower()]
    if sub_category:
        df = df[df["sub_category"].str.lower() == sub_category.lower()]

    df = df.dropna(subset=[col])
    df = df.sort_values(col, ascending=ascending).head(top_n)

    result_cols = [c for c in DISPLAY_COLS if c in df.columns]
    return df[result_cols].to_string(index=False)


@function_tool
def get_fund_details(scheme_name: str) -> str:
    """Get detailed information about a specific fund by name (fuzzy matched).

    Args:
        scheme_name: The name of the fund to look up (partial or full name).
    """
    df = _load_master()
    names = df["scheme_name"].dropna().tolist()
    matches = process.extract(scheme_name, names, scorer=fuzz.token_sort_ratio, limit=3)

    if not matches or matches[0][1] < 60:
        return f"No fund found matching '{scheme_name}'."

    results = []
    for match_name, score, _ in matches:
        row = df[df["scheme_name"] == match_name].iloc[0]
        info = {
            "scheme_name": row.get("scheme_name"),
            "amc": row.get("amc"),
            "category": row.get("category"),
            "sub_category": row.get("sub_category"),
            "benchmark": row.get("benchmark"),
            "riskometer": row.get("riskometer"),
            "nav_regular": row.get("nav_regular"),
            "return_1y": row.get("return_1y_regular"),
            "return_3y": row.get("return_3y_regular"),
            "return_5y": row.get("return_5y_regular"),
            "return_10y": row.get("return_10y_regular"),
            "aum_cr": row.get("aum_cr"),
            "trail_brokerage_pct": row.get("trail_brokerage_incl_gst"),
            "tieup_category": row.get("tieup_category"),
            "match_confidence": f"{score}%",
        }
        results.append("\n".join(f"  {k}: {v}" for k, v in info.items()))

    return "Top matches:\n\n" + "\n---\n".join(results)


@function_tool
def get_category_summary(risk_profile: str = "moderate") -> str:
    """Get summary statistics for each fund category and sub-category.
    Shows fund count, average score, average brokerage, average AUM, and average returns.

    Args:
        risk_profile: Risk profile: conservative, moderate, or aggressive.
    """
    df = _load_ranked()
    df = df[df["risk_profile"] == risk_profile]

    summary = df.groupby(["category", "sub_category"]).agg(
        fund_count=("scheme_name", "count"),
        avg_score=("composite_score", "mean"),
        avg_brokerage=("trail_brokerage_incl_gst", "mean"),
        avg_aum_cr=("aum_cr", "mean"),
        avg_return_1y=("return_1y_regular", "mean"),
        avg_return_3y=("return_3y_regular", "mean"),
    ).round(2).reset_index()

    summary = summary.sort_values("avg_score", ascending=False)
    return summary.to_string(index=False)


@function_tool
def get_brokerage_revenue_analysis(
    category: str = "",
    top_n: int = 15,
    risk_profile: str = "moderate",
) -> str:
    """Analyze brokerage revenue potential. Finds funds with high AUM AND high brokerage
    (best revenue potential), as well as high-AUM funds with low brokerage (missed revenue).

    Args:
        category: Filter by category. Empty string means all categories.
        top_n: Number of results per analysis (default 15).
        risk_profile: Risk profile: conservative, moderate, or aggressive.
    """
    df = _load_ranked()
    df = df[df["risk_profile"] == risk_profile].copy()
    if category:
        df = df[df["category"].str.lower() == category.lower()]

    df = df.dropna(subset=["trail_brokerage_incl_gst", "aum_cr"])
    df["estimated_annual_revenue_lakhs"] = (
        df["aum_cr"] * df["trail_brokerage_incl_gst"] / 100
    ).round(2)

    cols = ["scheme_name", "amc", "sub_category", "aum_cr",
            "trail_brokerage_incl_gst", "estimated_annual_revenue_lakhs",
            "composite_score", "tieup_category"]

    top_revenue = df.sort_values("estimated_annual_revenue_lakhs", ascending=False).head(top_n)
    high_aum_low_brok = df[
        (df["aum_cr"] > df["aum_cr"].quantile(0.75)) &
        (df["trail_brokerage_incl_gst"] < df["trail_brokerage_incl_gst"].quantile(0.25))
    ].sort_values("aum_cr", ascending=False).head(top_n)

    overall = (
        f"Total funds analyzed: {len(df)}\n"
        f"Total estimated annual revenue: {df['estimated_annual_revenue_lakhs'].sum():.2f} Cr\n"
        f"Avg brokerage: {df['trail_brokerage_incl_gst'].mean():.2f}%\n"
        f"Avg AUM: {df['aum_cr'].mean():.2f} Cr\n"
    )

    return (
        f"=== OVERVIEW ===\n{overall}\n"
        f"=== TOP REVENUE GENERATING FUNDS ===\n{top_revenue[cols].to_string(index=False)}\n\n"
        f"=== HIGH AUM BUT LOW BROKERAGE (MISSED REVENUE) ===\n{high_aum_low_brok[cols].to_string(index=False)}"
    )


@function_tool
def get_high_potential_opportunities(risk_profile: str = "moderate") -> str:
    """Find high-potential fund opportunities:
    1. High score + high brokerage (best to promote)
    2. High AUM + low score (shift candidates)
    3. A-tieup funds with high brokerage (strategic picks)

    Args:
        risk_profile: Risk profile: conservative, moderate, or aggressive.
    """
    df = _load_ranked()
    df = df[df["risk_profile"] == risk_profile].copy()
    df = df.dropna(subset=["trail_brokerage_incl_gst", "aum_cr", "composite_score"])

    cols = ["scheme_name", "amc", "sub_category", "composite_score",
            "trail_brokerage_incl_gst", "aum_cr", "tieup_category",
            "return_1y_regular", "return_3y_regular"]

    # Best to promote: top 25% score AND top 25% brokerage
    score_q75 = df["composite_score"].quantile(0.75)
    brok_q75 = df["trail_brokerage_incl_gst"].quantile(0.75)
    best_promote = df[
        (df["composite_score"] >= score_q75) &
        (df["trail_brokerage_incl_gst"] >= brok_q75)
    ].sort_values("composite_score", ascending=False).head(15)

    # High AUM but low score: shift candidates
    aum_q75 = df["aum_cr"].quantile(0.75)
    score_q25 = df["composite_score"].quantile(0.25)
    shift_candidates = df[
        (df["aum_cr"] >= aum_q75) &
        (df["composite_score"] <= score_q25)
    ].sort_values("aum_cr", ascending=False).head(10)

    # A-tieup with high brokerage
    strategic = df[
        (df["tieup_category"] == "A") &
        (df["trail_brokerage_incl_gst"] >= brok_q75)
    ].sort_values("composite_score", ascending=False).head(10)

    return (
        f"=== BEST FUNDS TO PROMOTE (High Score + High Brokerage) ===\n"
        f"{best_promote[cols].to_string(index=False)}\n\n"
        f"=== SHIFT CANDIDATES (High AUM but Low Score) ===\n"
        f"{shift_candidates[cols].to_string(index=False)}\n\n"
        f"=== STRATEGIC PICKS (A-TieUp + High Brokerage) ===\n"
        f"{strategic[cols].to_string(index=False)}"
    )


@function_tool
def search_funds(query: str, top_n: int = 10) -> str:
    """Search for funds by name using fuzzy matching.

    Args:
        query: Search term (fund name, AMC name, or keyword).
        top_n: Max results to return (default 10).
    """
    df = _load_master()
    names = df["scheme_name"].dropna().tolist()
    matches = process.extract(query, names, scorer=fuzz.token_sort_ratio, limit=top_n)

    if not matches:
        return f"No funds matching '{query}'."

    results = []
    for name, score, _ in matches:
        if score < 50:
            continue
        row = df[df["scheme_name"] == name].iloc[0]
        results.append({
            "scheme_name": name,
            "amc": row.get("amc"),
            "category": row.get("category"),
            "sub_category": row.get("sub_category"),
            "aum_cr": row.get("aum_cr"),
            "brokerage_pct": row.get("trail_brokerage_incl_gst"),
            "match_score": f"{score}%",
        })

    if not results:
        return f"No funds matching '{query}' with sufficient confidence."

    return pd.DataFrame(results).to_string(index=False)


@function_tool
def get_amc_analysis(risk_profile: str = "moderate") -> str:
    """Analyze AMC-level data: fund count, average brokerage, average score, AUM,
    and tie-up category per AMC.

    Args:
        risk_profile: Risk profile: conservative, moderate, or aggressive.
    """
    df = _load_ranked()
    df = df[df["risk_profile"] == risk_profile].copy()

    amc_summary = df.groupby("amc").agg(
        fund_count=("scheme_name", "count"),
        avg_score=("composite_score", "mean"),
        avg_brokerage=("trail_brokerage_incl_gst", "mean"),
        total_aum_cr=("aum_cr", "sum"),
        tieup=("tieup_category", "first"),
    ).round(2).reset_index()

    amc_summary = amc_summary.sort_values("total_aum_cr", ascending=False)
    return amc_summary.to_string(index=False)


@function_tool
def compare_funds(fund_names: str, risk_profile: str = "moderate") -> str:
    """Compare multiple funds side by side. Provide comma-separated fund names.

    Args:
        fund_names: Comma-separated fund names to compare (fuzzy matched).
        risk_profile: Risk profile: conservative, moderate, or aggressive.
    """
    df = _load_ranked()
    df = df[df["risk_profile"] == risk_profile]
    names_list = [n.strip() for n in fund_names.split(",") if n.strip()]

    all_names = df["scheme_name"].dropna().tolist()
    matched_rows = []

    for query_name in names_list:
        matches = process.extract(query_name, all_names, scorer=fuzz.token_sort_ratio, limit=1)
        if matches and matches[0][1] >= 60:
            matched = df[df["scheme_name"] == matches[0][0]].iloc[0]
            matched_rows.append(matched)

    if not matched_rows:
        return "Could not find any matching funds."

    result_df = pd.DataFrame(matched_rows)
    cols = [c for c in DISPLAY_COLS if c in result_df.columns]
    return result_df[cols].to_string(index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO HOLDINGS TOOLS (data comes from uploaded Scheme_wise_AUM via set_portfolio_data)
# ═══════════════════════════════════════════════════════════════════════════════

_portfolio_df = None


def parse_portfolio_excel(file_bytes: bytes) -> pd.DataFrame:
    """Parse a Scheme_wise_AUM .xls/.xlsx file into a clean DataFrame."""
    import io
    raw = pd.read_excel(io.BytesIO(file_bytes), header=None)
    # Row 0 has main headers, row 1 has sub-headers (EQUITY, DEBT, etc.), data from row 2+
    df = raw.iloc[2:].reset_index(drop=True)
    df.columns = ["sr_no", "amc", "scheme", "nav", "equity", "debt",
                   "hybrid", "physical_assets", "others", "total"]
    df = df.dropna(subset=["scheme"])
    df = df[df["scheme"].astype(str).str.strip() != ""]
    for c in ["nav", "equity", "debt", "hybrid", "physical_assets", "others", "total"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def set_portfolio_data(portfolio_df: pd.DataFrame | None):
    """Called from the Streamlit app to inject uploaded portfolio data into the tools."""
    global _portfolio_df
    _portfolio_df = portfolio_df


def _get_portfolio() -> pd.DataFrame | None:
    """Return uploaded portfolio, or fallback to default Scheme_wise_AUM.xls on disk."""
    if _portfolio_df is not None:
        return _portfolio_df

    # Fallback: load from default file (same as what the dashboard page does)
    default_file = os.path.join(DATA_DIR, "..", "..", "Scheme_wise_AUM.xls")
    if os.path.exists(default_file):
        from analysis.portfolio_review import load_aum_data
        return load_aum_data()
    return None


def _no_portfolio_msg() -> str:
    return (
        "No portfolio data available. The default Scheme_wise_AUM.xls was not found "
        "and no file was uploaded. Please upload your Scheme_wise_AUM file."
    )


@function_tool
def get_portfolio_holdings(top_n: int = 20, sort_by: str = "total") -> str:
    """Get current portfolio holdings (from uploaded Scheme_wise_AUM). Shows schemes held,
    their AUM broken down by asset class (equity, debt, hybrid, etc).

    Args:
        top_n: Number of holdings to return (default 20, max 50).
        sort_by: Sort by 'total', 'equity', 'debt', or 'hybrid'. Default 'total'.
    """
    df = _get_portfolio()
    if df is None or df.empty:
        return _no_portfolio_msg()

    top_n = min(top_n, 50)
    sort_col = sort_by if sort_by in df.columns else "total"
    result = df.sort_values(sort_col, ascending=False).head(top_n)

    out = result[["scheme", "amc", "equity", "debt", "hybrid", "total"]].copy()
    for c in ["equity", "debt", "hybrid", "total"]:
        out[f"{c}_lakh"] = (out[c] / 1e5).round(1)

    display = out[["scheme", "amc", "equity_lakh", "debt_lakh", "hybrid_lakh", "total_lakh"]]
    display = display.rename(columns={
        "equity_lakh": "Equity(L)", "debt_lakh": "Debt(L)",
        "hybrid_lakh": "Hybrid(L)", "total_lakh": "Total(L)",
    })

    total_aum = df["total"].sum()
    return (
        f"=== PORTFOLIO HOLDINGS (Total AUM: Rs {total_aum / 1e7:.2f} Cr, "
        f"{len(df)} schemes) ===\n{display.to_string(index=False)}"
    )


@function_tool
def get_portfolio_concentration() -> str:
    """Analyze portfolio concentration by AMC and asset class.
    Shows which AMCs dominate the portfolio and the equity/debt/hybrid split."""
    df = _get_portfolio()
    if df is None or df.empty:
        return _no_portfolio_msg()

    total_aum = df["total"].sum()

    # AMC concentration
    amc_aum = df.groupby("amc")["total"].sum().sort_values(ascending=False).reset_index()
    amc_aum["pct"] = (amc_aum["total"] / total_aum * 100).round(1)
    amc_aum["aum_lakh"] = (amc_aum["total"] / 1e5).round(1)

    # Asset class split
    equity_total = df["equity"].sum()
    debt_total = df["debt"].sum()
    hybrid_total = df["hybrid"].sum()
    other_total = df["physical_assets"].sum() + df["others"].sum()

    return (
        f"=== PORTFOLIO CONCENTRATION ===\n"
        f"Total AUM: Rs {total_aum / 1e7:.2f} Cr across {len(df)} schemes\n\n"
        f"ASSET CLASS SPLIT:\n"
        f"  Equity: Rs {equity_total / 1e7:.2f} Cr ({equity_total / total_aum * 100:.1f}%)\n"
        f"  Debt:   Rs {debt_total / 1e7:.2f} Cr ({debt_total / total_aum * 100:.1f}%)\n"
        f"  Hybrid: Rs {hybrid_total / 1e7:.2f} Cr ({hybrid_total / total_aum * 100:.1f}%)\n"
        f"  Other:  Rs {other_total / 1e7:.2f} Cr ({other_total / total_aum * 100:.1f}%)\n\n"
        f"AMC CONCENTRATION (Top 15):\n"
        f"{amc_aum[['amc', 'aum_lakh', 'pct']].head(15).rename(columns={'aum_lakh': 'AUM(L)', 'pct': '%'}).to_string(index=False)}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLIENT INSIGHTS TOOLS (data comes from uploaded reports via set_client_data)
# ═══════════════════════════════════════════════════════════════════════════════

def _no_client_data_msg() -> str:
    return (
        "No client data available. The advisor needs to upload the Business Insight Report "
        "and Live SIP Report in the 'Client Insights' page first, then come back here."
    )


@function_tool
def get_client_overview() -> str:
    """Get an overview of the client base: total clients, total AUM, total SIP,
    estimated revenue, and gap counts. Requires uploaded client data."""
    m = _client_data.get("metrics")
    if not m:
        return _no_client_data_msg()

    return (
        f"=== CLIENT BASE OVERVIEW ===\n"
        f"Total clients: {m.get('total_clients', 0)}\n"
        f"Total MF AUM: Rs {m.get('total_aum', 0) / 1e7:.2f} Cr\n"
        f"Total Live SIP Amount: Rs {m.get('live_sip_amount', 0) / 1e5:.2f} Lakh/month\n"
        f"Estimated Annual Revenue (1.2% trail): Rs {m.get('est_annual_revenue', 0) / 1e5:.2f} Lakh\n\n"
        f"=== GAP ANALYSIS SUMMARY ===\n"
        f"High AUM but No SIP: {m.get('high_aum_no_sip_count', 0)} clients\n"
        f"Reduced SIP (declined): {m.get('reduced_sip_count', 0)} clients\n"
        f"No Top-Up registered: {m.get('no_topup_count', 0)} clients\n"
        f"SIP Terminated: {m.get('sip_terminated_count', 0)} clients\n"
        f"Below Benchmark (AUM/SIP ratio high): {m.get('below_benchmark_count', 0)} clients\n"
    )


@function_tool
def get_top_clients(metric: str = "aum", top_n: int = 15, ascending: bool = False) -> str:
    """Get top clients ranked by AUM or SIP amount.

    Args:
        metric: Rank by 'aum' (Total MF AUM) or 'sip' (Live SIP Amount). Default 'aum'.
        top_n: Number of clients to return (default 15, max 30).
        ascending: If True, sort ascending (lowest first). Default False.
    """
    df = _client_data.get("business_df")
    if df is None or df.empty:
        return _no_client_data_msg()

    top_n = min(top_n, 30)
    col = "Total_MF_AUM" if metric == "aum" else "Live_SIP_Amount"
    if col not in df.columns:
        return f"Column '{col}' not found in client data."

    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    result = df.sort_values(col, ascending=ascending).head(top_n)

    display_cols = ["Group"]
    for c in ["Total_MF_AUM", "Live_SIP_Amount", "Est_Annual_Revenue",
              "SIP_AUM", "Total_SIP_Amount", "SIP_Schemes_Count", "Client_Tier"]:
        if c in result.columns:
            display_cols.append(c)

    out = result[display_cols].copy()
    if "Total_MF_AUM" in out.columns:
        out["Total_MF_AUM_Lakh"] = (out["Total_MF_AUM"] / 1e5).round(1)
    if "Live_SIP_Amount" in out.columns:
        out["Live_SIP_K"] = (out["Live_SIP_Amount"] / 1e3).round(1)

    return out.to_string(index=False)


@function_tool
def get_client_gaps(gap_type: str = "all", top_n: int = 20) -> str:
    """Get clients falling into specific gap categories for targeted action.

    Args:
        gap_type: One of: high_aum_no_sip, reduced_sip, no_topup, sip_terminated, below_benchmark, or 'all' for summary.
        top_n: Max clients to show per gap (default 20).
    """
    gaps = _client_data.get("gaps")
    df = _client_data.get("business_df")
    if gaps is None or df is None:
        return _no_client_data_msg()

    if gap_type == "all":
        lines = []
        for gtype, gdf in gaps.items():
            count = len(gdf) if isinstance(gdf, pd.DataFrame) else 0
            lines.append(f"  {gtype}: {count} clients")
        return "=== ALL GAP TYPES ===\n" + "\n".join(lines)

    gap_df = gaps.get(gap_type, pd.DataFrame())
    if not isinstance(gap_df, pd.DataFrame) or gap_df.empty:
        return f"No clients found for gap type '{gap_type}'."

    display_cols = ["Group"]
    for c in ["Total_MF_AUM", "Live_SIP_Amount", "SIP_Change_2Yrs",
              "TopUp_SIP_Amount", "SIP_Closed"]:
        if c in gap_df.columns:
            display_cols.append(c)

    out = gap_df[display_cols].head(top_n).copy()
    if "Total_MF_AUM" in out.columns:
        out["AUM_Lakh"] = (pd.to_numeric(out["Total_MF_AUM"], errors="coerce").fillna(0) / 1e5).round(1)

    return f"=== {gap_type.upper()} ({len(gap_df)} clients total, showing top {min(top_n, len(gap_df))}) ===\n{out.to_string(index=False)}"


@function_tool
def get_pareto_analysis() -> str:
    """Get Pareto (80/20) analysis of the client base — shows how much AUM the top 20%
    of clients contribute. Requires uploaded client data."""
    pareto = _client_data.get("pareto")
    if not pareto:
        return _no_client_data_msg()

    return (
        f"=== PARETO ANALYSIS ===\n"
        f"Total clients (with AUM > 0): {pareto.get('total_clients', 0)}\n"
        f"Total AUM: Rs {pareto.get('total_aum', 0) / 1e7:.2f} Cr\n"
        f"Top 20% clients count: {pareto.get('top_20_pct_clients', 0)}\n"
        f"Top 20% clients AUM: Rs {pareto.get('top_20_aum', 0) / 1e7:.2f} Cr\n"
        f"Top 20% contribute: {pareto.get('top_20_aum_pct', 0):.1f}% of total AUM\n"
    )


@function_tool
def get_sip_leakage_analysis() -> str:
    """Analyze where SIP inflows are being lost — clients with high AUM but no SIP,
    terminated SIPs, and reduced SIPs. Shows the revenue impact. Requires uploaded client data."""
    gaps = _client_data.get("gaps")
    df = _client_data.get("business_df")
    if gaps is None or df is None:
        return _no_client_data_msg()

    lines = ["=== SIP LEAKAGE ANALYSIS ===\n"]

    # High AUM no SIP
    g1 = gaps.get("high_aum_no_sip", pd.DataFrame())
    if not g1.empty and "Total_MF_AUM" in g1.columns:
        total_aum = pd.to_numeric(g1["Total_MF_AUM"], errors="coerce").fillna(0).sum()
        lines.append(f"HIGH AUM / NO SIP: {len(g1)} clients, AUM = Rs {total_aum / 1e7:.2f} Cr")
        lines.append(f"  Potential SIP revenue if converted (est 1.2%): Rs {total_aum * 0.012 / 1e5:.2f} Lakh/year")
        top5 = g1.nlargest(5, "Total_MF_AUM")[["Group", "Total_MF_AUM"]].copy()
        top5["AUM_Lakh"] = (top5["Total_MF_AUM"] / 1e5).round(1)
        lines.append(f"  Top 5 clients:\n{top5[['Group', 'AUM_Lakh']].to_string(index=False)}\n")

    # Reduced SIP
    g2 = gaps.get("reduced_sip", pd.DataFrame())
    if not g2.empty:
        lines.append(f"REDUCED SIP: {len(g2)} clients have declining SIP amounts")
        if "SIP_Change_2Yrs" in g2.columns:
            total_decline = pd.to_numeric(g2["SIP_Change_2Yrs"], errors="coerce").fillna(0).sum()
            lines.append(f"  Total SIP decline over 2 years: Rs {abs(total_decline) / 1e3:.1f}K\n")

    # Terminated SIP
    g3 = gaps.get("sip_terminated", pd.DataFrame())
    if not g3.empty:
        lines.append(f"SIP TERMINATED: {len(g3)} clients have closed SIPs")
        if "SIP_Closed" in g3.columns:
            total_closed = pd.to_numeric(g3["SIP_Closed"], errors="coerce").fillna(0).sum()
            lines.append(f"  Total SIPs closed: {int(total_closed)}\n")

    # No top-up
    g4 = gaps.get("no_topup", pd.DataFrame())
    if not g4.empty:
        lines.append(f"NO TOP-UP: {len(g4)} active SIP clients have not registered a top-up\n")

    if len(lines) == 1:
        return "No SIP leakage data found. Make sure client reports have been uploaded."

    return "\n".join(lines)


@function_tool
def get_client_growth_segments() -> str:
    """Identify client segments with the highest growth potential based on AUM tiers
    and SIP activity. Requires uploaded client data."""
    df = _client_data.get("business_df")
    if df is None or df.empty:
        return _no_client_data_msg()

    df = df.copy()
    if "Total_MF_AUM" not in df.columns:
        return "Total_MF_AUM column not found in client data."

    df["Total_MF_AUM"] = pd.to_numeric(df["Total_MF_AUM"], errors="coerce").fillna(0)

    # Tier classification
    def tier(aum):
        if aum >= 2e7:
            return "Platinum (>2Cr)"
        elif aum >= 1e7:
            return "Gold (1-2Cr)"
        elif aum >= 5e6:
            return "Silver (50L-1Cr)"
        elif aum >= 1e6:
            return "Bronze (10-50L)"
        else:
            return "Starter (<10L)"

    df["Tier"] = df["Total_MF_AUM"].apply(tier)

    agg = {"Group": "count", "Total_MF_AUM": ["sum", "mean"]}
    if "Live_SIP_Amount" in df.columns:
        df["Live_SIP_Amount"] = pd.to_numeric(df["Live_SIP_Amount"], errors="coerce").fillna(0)
        agg["Live_SIP_Amount"] = ["sum", "mean"]

    summary = df.groupby("Tier").agg(agg)
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.rename(columns={
        "Group_count": "clients",
        "Total_MF_AUM_sum": "total_aum",
        "Total_MF_AUM_mean": "avg_aum",
    })

    # Format
    if "total_aum" in summary.columns:
        summary["total_aum_Cr"] = (summary["total_aum"] / 1e7).round(2)
        summary["avg_aum_Lakh"] = (summary["avg_aum"] / 1e5).round(1)
    if "Live_SIP_Amount_sum" in summary.columns:
        summary["total_sip_K"] = (summary["Live_SIP_Amount_sum"] / 1e3).round(1)
        summary["avg_sip"] = summary["Live_SIP_Amount_mean"].round(0)

    # Identify growth potential
    display = summary[["clients", "total_aum_Cr", "avg_aum_Lakh"]].copy()
    if "total_sip_K" in summary.columns:
        display["total_sip_K"] = summary["total_sip_K"]
        display["avg_sip"] = summary["avg_sip"]

    # Clients without SIP in each tier
    no_sip_counts = []
    for t in display.index:
        tier_df = df[df["Tier"] == t]
        if "Live_SIP_Amount" in tier_df.columns:
            no_sip = (tier_df["Live_SIP_Amount"] == 0).sum()
        else:
            no_sip = 0
        no_sip_counts.append(no_sip)
    display["no_sip_clients"] = no_sip_counts

    return f"=== CLIENT GROWTH SEGMENTS ===\n{display.to_string()}"


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD PAGE TOOLS (wrapping analysis modules)
# ═══════════════════════════════════════════════════════════════════════════════

@function_tool
def get_peer_comparison(fund_names: str, risk_profile: str = "moderate") -> str:
    """Get peer comparison for one or more funds. Shows all funds in the same
    sub-category alongside the selected funds, with scores, returns, brokerage.

    Args:
        fund_names: Comma-separated fund names to compare with peers (fuzzy matched).
        risk_profile: Risk profile: conservative, moderate, or aggressive.
    """
    from analysis.peer_comparison import get_peer_comparison as _get_peers

    df = _load_ranked()
    names_list = [n.strip() for n in fund_names.split(",") if n.strip()]
    all_names = df["scheme_name"].dropna().unique().tolist()

    matched = []
    for q in names_list:
        m = process.extractOne(q, all_names, scorer=fuzz.token_sort_ratio, limit=1)
        if m and m[1] >= 60:
            matched.append(m[0])

    if not matched:
        return "Could not find any matching funds."

    peers = _get_peers(matched, risk_profile, df=df)
    if peers is None or peers.empty:
        return "No peers found for the selected funds."

    cols = ["scheme_name", "sub_category", "composite_score", "rank",
            "return_1y_regular", "return_3y_regular", "return_5y_regular",
            "trail_brokerage_incl_gst", "aum_cr", "tieup_category", "is_selected"]
    cols = [c for c in cols if c in peers.columns]
    peers = peers.sort_values(["sub_category", "rank"])

    selected_count = peers["is_selected"].sum() if "is_selected" in peers.columns else 0
    total = len(peers)
    return (
        f"=== PEER COMPARISON ({selected_count} selected, {total} total peers) ===\n"
        f"{peers[cols].head(30).to_string(index=False)}"
    )


@function_tool
def get_fund_shift_alternatives(scheme_name: str, risk_profile: str = "moderate", n: int = 5) -> str:
    """Find better alternative funds to shift into from a given fund.
    Alternatives are in the same sub-category, have brokerage >= current fund,
    and are in the top 50% by composite score.

    Args:
        scheme_name: Fund name to find alternatives for (fuzzy matched).
        risk_profile: Risk profile: conservative, moderate, or aggressive.
        n: Number of alternatives to return (default 5, max 10).
    """
    from analysis.fund_shift import suggest_alternatives

    df = _load_ranked()
    all_names = df[df["risk_profile"] == risk_profile]["scheme_name"].dropna().unique().tolist()
    m = process.extractOne(scheme_name, all_names, scorer=fuzz.token_sort_ratio)
    if not m or m[1] < 60:
        return f"Could not find fund matching '{scheme_name}'."

    exact_name = m[0]
    n = min(n, 10)
    alts = suggest_alternatives(exact_name, risk_profile, n=n, df=df)
    if alts is None or alts.empty:
        return f"No better alternatives found for '{exact_name}'."

    # Show the current fund info + alternatives
    current = df[(df["scheme_name"] == exact_name) & (df["risk_profile"] == risk_profile)].iloc[0]
    header = (
        f"CURRENT FUND: {exact_name}\n"
        f"  Score: {current.get('composite_score', 'N/A')}, "
        f"Brokerage: {current.get('trail_brokerage_incl_gst', 'N/A')}%, "
        f"1Y Return: {current.get('return_1y_regular', 'N/A')}%, "
        f"AAUM: {current.get('aum_cr', 'N/A')} Cr\n\n"
    )

    cols = ["scheme_name", "composite_score", "trail_brokerage_incl_gst",
            "return_1y_regular", "return_3y_regular", "aum_cr", "tieup_category",
            "delta_composite", "delta_brokerage", "delta_return_1y"]
    cols = [c for c in cols if c in alts.columns]

    return header + f"ALTERNATIVES ({len(alts)} found):\n{alts[cols].to_string(index=False)}"


@function_tool
def review_portfolio_exposure(risk_profile: str = "moderate", aum_threshold: str = "50 Lakh") -> str:
    """Review current portfolio holdings and flag underperforming schemes.
    Flags schemes with high AAUM but low composite score or low brokerage.
    Requires uploaded Scheme_wise_AUM data.

    Args:
        risk_profile: Risk profile: conservative, moderate, or aggressive.
        aum_threshold: Minimum AAUM threshold. One of: '25 Lakh', '50 Lakh', '75 Lakh', '1 Cr', '1.5 Cr', '2 Cr'.
    """
    from analysis.portfolio_review import flag_underperforming_schemes, AUM_THRESHOLDS

    portfolio_df = _get_portfolio()
    if portfolio_df is None:
        return _no_portfolio_msg()

    aum_df = portfolio_df.copy()
    ranked_df = _load_ranked()
    thresh = AUM_THRESHOLDS.get(aum_threshold, 5000000)

    result = flag_underperforming_schemes(
        aum_df, ranked_df,
        risk_profile=risk_profile,
        aum_threshold=thresh,
        include_brokerage_flag=True,
    )

    summary = result["summary"]
    flagged = result.get("flagged", pd.DataFrame())
    aum_by_asset = result.get("aum_by_asset", {})

    lines = [
        f"=== PORTFOLIO EXPOSURE REVIEW ===",
        f"Total AAUM: Rs {summary.get('total_aum', 0) / 1e7:.2f} Cr",
        f"Total Schemes: {summary.get('total_schemes', 0)}",
        f"Matched to Ranked: {summary.get('matched_schemes', 0)}",
        f"Above Threshold ({aum_threshold}): {summary.get('schemes_above_threshold', 0)}",
        f"Flagged (underperforming): {summary.get('flagged_count', 0)}",
        "",
    ]

    if aum_by_asset:
        lines.append("ASSET CLASS BREAKDOWN:")
        for asset, val in sorted(aum_by_asset.items(), key=lambda x: -x[1]):
            lines.append(f"  {asset}: Rs {val / 1e7:.2f} Cr")
        lines.append("")

    if not flagged.empty:
        cols = ["scheme", "total", "composite_score", "rank",
                "trail_brokerage_incl_gst", "flag_reason"]
        cols = [c for c in cols if c in flagged.columns]
        flagged_display = flagged[cols].head(20).copy()
        if "total" in flagged_display.columns:
            flagged_display["total_lakh"] = (flagged_display["total"] / 1e5).round(1)
        lines.append(f"FLAGGED SCHEMES ({len(flagged)} total, showing top 20):")
        lines.append(flagged_display.to_string(index=False))
    else:
        lines.append("No schemes flagged — portfolio looks healthy!")

    return "\n".join(lines)


@function_tool
def quantify_portfolio_exposure(risk_profile: str = "moderate") -> str:
    """Quantify how over-exposed or under-utilized each scheme in the portfolio is.
    Compares each scheme's portfolio weight (Exposure %) against its quality
    (composite score percentile within sub-category) to compute an Exposure Gap.

    Categories:
    - OVEREXPOSED: High portfolio weight but low quality — reduce/shift
    - UNDERUTILIZED: Low portfolio weight but high quality — increase allocation
    - WELL-BALANCED: Exposure roughly matches quality — hold
    - DEAD WEIGHT: Low weight and low quality — consider exiting

    Requires uploaded Scheme_wise_AUM data.

    Args:
        risk_profile: Risk profile: conservative, moderate, or aggressive.
    """
    from analysis.portfolio_review import flag_underperforming_schemes

    portfolio_df = _get_portfolio()
    if portfolio_df is None:
        return _no_portfolio_msg()

    ranked_df = _load_ranked()
    result = flag_underperforming_schemes(
        portfolio_df, ranked_df,
        risk_profile=risk_profile,
        aum_threshold=0,  # include all schemes
        include_brokerage_flag=True,
    )

    holdings = result.get("all_holdings", pd.DataFrame())
    if holdings.empty or "composite_score" not in holdings.columns:
        return "Could not merge portfolio with ranked data. Check uploaded file."

    h = holdings.dropna(subset=["composite_score"]).copy()
    if h.empty:
        return "No schemes in portfolio matched the ranked fund database."

    total_aum = h["total"].sum()
    if total_aum == 0:
        return "Total portfolio AUM is zero."

    # Exposure %: scheme weight in portfolio
    h["exposure_pct"] = (h["total"] / total_aum * 100).round(2)

    # Quality %: composite score percentile within sub-category (0-100)
    h["quality_pct"] = h.groupby("sub_category")["composite_score"].rank(pct=True).mul(100).round(1)

    # Exposure Gap = Exposure % - normalized quality
    # Normalize quality to same scale as exposure for fair comparison
    max_exposure = h["exposure_pct"].max()
    h["quality_norm"] = (h["quality_pct"] / 100 * max_exposure).round(2)
    h["exposure_gap"] = (h["exposure_pct"] - h["quality_norm"]).round(2)

    # Categorize
    median_exposure = h["exposure_pct"].median()
    median_quality = h["quality_pct"].median()

    def categorize(row):
        high_exp = row["exposure_pct"] >= median_exposure
        high_qual = row["quality_pct"] >= median_quality
        if high_exp and not high_qual:
            return "OVEREXPOSED"
        elif not high_exp and high_qual:
            return "UNDERUTILIZED"
        elif high_exp and high_qual:
            return "WELL-BALANCED"
        else:
            return "DEAD WEIGHT"

    h["category"] = h.apply(categorize, axis=1)

    cols = ["scheme", "exposure_pct", "quality_pct", "exposure_gap",
            "composite_score", "rank", "trail_brokerage_incl_gst", "category"]
    cols = [c for c in cols if c in h.columns]
    h["aum_lakh"] = (h["total"] / 1e5).round(1)

    # Summary stats
    overexposed = h[h["category"] == "OVEREXPOSED"]
    underutilized = h[h["category"] == "UNDERUTILIZED"]
    dead_weight = h[h["category"] == "DEAD WEIGHT"]
    well_balanced = h[h["category"] == "WELL-BALANCED"]

    overexposed_aum = overexposed["total"].sum()
    underutilized_aum = underutilized["total"].sum()
    dead_weight_aum = dead_weight["total"].sum()

    # Potential brokerage gain if rebalanced
    if not overexposed.empty and not underutilized.empty:
        avg_brok_over = overexposed["trail_brokerage_incl_gst"].mean()
        avg_brok_under = underutilized["trail_brokerage_incl_gst"].mean()
        brok_gain = (avg_brok_under - avg_brok_over) if pd.notna(avg_brok_under) and pd.notna(avg_brok_over) else 0
    else:
        avg_brok_over = avg_brok_under = brok_gain = 0

    lines = [
        f"=== PORTFOLIO EXPOSURE QUANTIFICATION ===",
        f"Total Portfolio: Rs {total_aum / 1e7:.2f} Cr | {len(h)} matched schemes\n",
        f"SUMMARY:",
        f"  OVEREXPOSED:   {len(overexposed)} schemes, Rs {overexposed_aum / 1e7:.2f} Cr "
        f"(high weight, low quality — reduce)",
        f"  UNDERUTILIZED: {len(underutilized)} schemes, Rs {underutilized_aum / 1e7:.2f} Cr "
        f"(low weight, high quality — increase)",
        f"  WELL-BALANCED: {len(well_balanced)} schemes (hold)",
        f"  DEAD WEIGHT:   {len(dead_weight)} schemes, Rs {dead_weight_aum / 1e7:.2f} Cr "
        f"(low weight, low quality — consider exiting)",
        "",
    ]

    if brok_gain > 0:
        potential = overexposed_aum * brok_gain / 100
        lines.append(
            f"  Potential brokerage uplift if shifted from overexposed (avg {avg_brok_over:.2f}%) "
            f"to underutilized (avg {avg_brok_under:.2f}%): ~Rs {potential / 1e5:.1f} Lakh/year\n"
        )

    if not overexposed.empty:
        top_over = overexposed.sort_values("exposure_gap", ascending=False).head(10)
        lines.append(f"TOP OVEREXPOSED (reduce these):")
        lines.append(top_over[cols + ["aum_lakh"]].to_string(index=False))
        lines.append("")

    if not underutilized.empty:
        top_under = underutilized.sort_values("quality_pct", ascending=False).head(10)
        lines.append(f"TOP UNDERUTILIZED (increase these):")
        lines.append(top_under[cols + ["aum_lakh"]].to_string(index=False))
        lines.append("")

    if not dead_weight.empty:
        lines.append(f"DEAD WEIGHT (consider exiting):")
        lines.append(dead_weight[cols + ["aum_lakh"]].to_string(index=False))

    return "\n".join(lines)


@function_tool
def get_portfolio_alternatives(risk_profile: str = "moderate", aum_threshold: str = "50 Lakh") -> str:
    """Get better alternative funds for flagged underperforming schemes in the portfolio.
    Requires uploaded Scheme_wise_AUM data.

    Args:
        risk_profile: Risk profile: conservative, moderate, or aggressive.
        aum_threshold: Minimum AAUM threshold. One of: '25 Lakh', '50 Lakh', '75 Lakh', '1 Cr', '1.5 Cr', '2 Cr'.
    """
    from analysis.portfolio_review import flag_underperforming_schemes, get_alternatives_for_flagged, AUM_THRESHOLDS

    portfolio_df = _get_portfolio()
    if portfolio_df is None:
        return _no_portfolio_msg()

    ranked_df = _load_ranked()
    thresh = AUM_THRESHOLDS.get(aum_threshold, 5000000)

    result = flag_underperforming_schemes(
        portfolio_df, ranked_df,
        risk_profile=risk_profile,
        aum_threshold=thresh,
        include_brokerage_flag=True,
    )
    flagged = result.get("flagged", pd.DataFrame())
    if flagged.empty:
        return "No underperforming schemes found — no alternatives needed."

    alts = get_alternatives_for_flagged(flagged, ranked_df, risk_profile=risk_profile, n=3)
    if alts is None or alts.empty:
        return "Flagged schemes found but no better alternatives available."

    cols = ["flagged_scheme", "flagged_aum", "flagged_score", "flagged_brokerage",
            "alternative_scheme", "alternative_score", "alternative_brokerage",
            "alternative_return_1y", "alternative_aum", "sub_category"]
    cols = [c for c in cols if c in alts.columns]

    return (
        f"=== ALTERNATIVES FOR {len(flagged)} FLAGGED SCHEMES ===\n"
        f"{alts[cols].head(30).to_string(index=False)}"
    )


@function_tool
def get_amc_concentration_analysis(
    risk_profile: str = "moderate",
    top_n_per_subcat: int = 5,
    category: str = "",
) -> str:
    """Analyze AMC concentration among top-ranked funds. Flags AMCs exceeding 30%
    concentration and suggests rebalancing from under-represented AMCs.

    Args:
        risk_profile: Risk profile: conservative, moderate, or aggressive.
        top_n_per_subcat: Top N funds per sub-category to analyze (default 5).
        category: Filter by category (empty = all).
    """
    from analysis.amc_concentration import compute_amc_concentration

    df = _load_ranked()
    categories = [category] if category else None

    result = compute_amc_concentration(
        risk_profile=risk_profile,
        top_n_per_subcat=top_n_per_subcat,
        categories=categories,
        df=df,
    )

    summary = result.get("summary", pd.DataFrame())
    alerts = result.get("alerts", [])
    rebalance = result.get("rebalance", pd.DataFrame())
    total = result.get("total_funds_analysed", 0)

    lines = [f"=== AMC CONCENTRATION (Top {top_n_per_subcat}/sub-cat, {total} funds) ===\n"]

    if alerts:
        lines.append(f"ALERTS (>30% concentration): {', '.join(alerts)}\n")

    if not summary.empty:
        cols = [c for c in ["amc", "count", "pct", "avg_composite", "tieup_category", "alert"] if c in summary.columns]
        lines.append(f"AMC DISTRIBUTION:\n{summary[cols].head(20).to_string(index=False)}\n")

    if not rebalance.empty:
        cols = [c for c in ["amc", "count", "avg_composite", "tieup_category"] if c in rebalance.columns]
        lines.append(f"REBALANCE CANDIDATES (under-represented AMCs):\n{rebalance[cols].head(10).to_string(index=False)}")

    return "\n".join(lines)


@function_tool
def get_holdings_amc_concentration(aum_threshold: str = "25 Lakh") -> str:
    """Analyze AMC concentration in current portfolio holdings (from uploaded Scheme_wise_AUM).
    Shows which AMCs dominate and flags any with >30% share.

    Args:
        aum_threshold: Minimum AAUM threshold. One of: '25 Lakh', '50 Lakh', '75 Lakh', '1 Cr', '1.5 Cr', '2 Cr'.
    """
    from analysis.amc_concentration import compute_current_amc_concentration
    from analysis.portfolio_review import AUM_THRESHOLDS

    portfolio_df = _get_portfolio()
    if portfolio_df is None:
        return _no_portfolio_msg()

    thresh = AUM_THRESHOLDS.get(aum_threshold, 2500000)
    result = compute_current_amc_concentration(aum_threshold=thresh)

    summary = result.get("summary", pd.DataFrame())
    total_aum = result.get("total_aum", 0)
    total_amcs = result.get("total_amcs", 0)

    lines = [
        f"=== HOLDINGS AMC CONCENTRATION ===",
        f"Total AAUM: Rs {total_aum / 1e7:.2f} Cr | AMCs: {total_amcs}\n",
    ]

    if not summary.empty:
        cols = [c for c in ["amc", "aum", "pct", "alert"] if c in summary.columns]
        display = summary[cols].head(20).copy()
        if "aum" in display.columns:
            display["aum_lakh"] = (display["aum"] / 1e5).round(1)
        lines.append(display.to_string(index=False))

    alerts = summary[summary.get("alert", pd.Series(dtype=str)) == "⚠️"] if "alert" in summary.columns else pd.DataFrame()
    if not alerts.empty:
        lines.append(f"\n⚠️ AMCs exceeding 30%: {', '.join(alerts['amc'].tolist())}")

    return "\n".join(lines)


@function_tool
def build_recommended_portfolio(
    basket_name: str = "Balanced - Equity 50",
    risk_profile: str = "moderate",
) -> str:
    """Build a recommended model portfolio from pre-defined baskets.
    Selects the top-ranked fund per sub-category with AMC diversification (max 30% per AMC).

    Available baskets: Conservative - Equity 30, Conservative - Equity 20,
    Balanced - Equity 40, Balanced - Equity 50, Balanced - Equity 60,
    Growth - Equity 70, Growth - Equity 80, Growth - Equity 90,
    SIP - Diversified, SIP - Equity Heavy, SIP - Debt Heavy,
    Tax Saving (ELSS), Retirement - Conservative, Retirement - Moderate,
    Child Education - Growth, Child Education - Balanced.

    Args:
        basket_name: Portfolio basket name from the list above.
        risk_profile: Risk profile: conservative, moderate, or aggressive.
    """
    from analysis.portfolio_builder import build_portfolio, get_portfolio_stats, BASKET_NAMES

    # Fuzzy match basket name
    matched = process.extractOne(basket_name, BASKET_NAMES, scorer=fuzz.token_sort_ratio)
    if not matched or matched[1] < 60:
        return f"Unknown basket '{basket_name}'. Available: {', '.join(BASKET_NAMES)}"

    basket = matched[0]
    df = _load_ranked()
    result = build_portfolio(ranked_df=df, basket_name=basket, risk_profile=risk_profile)

    portfolio = result.get("portfolio", pd.DataFrame())
    basket_info = result.get("basket", {})
    swaps = result.get("swaps", [])
    missing = result.get("missing", [])

    if portfolio.empty:
        return f"Could not build portfolio for '{basket}'."

    stats = get_portfolio_stats(portfolio)

    lines = [
        f"=== RECOMMENDED PORTFOLIO: {basket} ({risk_profile}) ===",
        f"Risk Level: {basket_info.get('risk_level', 'N/A')}",
        f"Allocation: Equity {basket_info.get('equity_pct', 0)}% | "
        f"Debt {basket_info.get('debt_pct', 0)}% | "
        f"Hybrid {basket_info.get('hybrid_pct', 0)}%\n",
        f"STATS:",
        f"  Schemes: {stats.get('num_schemes', 0)} | AMCs: {stats.get('num_amcs', 0)}",
        f"  Weighted Avg Brokerage: {stats.get('weighted_avg_brok', 0):.2f}%",
        f"  Weighted Avg 1Y Return: {stats.get('weighted_avg_ret1y', 0):.2f}%",
        f"  Avg Composite Score: {stats.get('avg_score', 0):.1f}",
        f"  TieUp: A={stats.get('tieup_a_count', 0)} | B={stats.get('tieup_b_count', 0)} | None={stats.get('no_tieup_count', 0)}\n",
    ]

    cols = ["scheme_name", "sub_category", "allocation_pct", "composite_score",
            "trail_brokerage_incl_gst", "return_1y_regular", "aum_cr", "tieup_category"]
    cols = [c for c in cols if c in portfolio.columns]
    lines.append(f"PORTFOLIO ({len(portfolio)} funds):")
    lines.append(portfolio[cols].to_string(index=False))

    if swaps:
        lines.append(f"\nAMC DIVERSIFICATION SWAPS:\n" + "\n".join(f"  - {s}" for s in swaps))
    if missing:
        lines.append(f"\nMISSING SUB-CATEGORIES: {', '.join(missing)}")

    return "\n".join(lines)


@function_tool
def list_portfolio_baskets() -> str:
    """List all available pre-built portfolio baskets with their allocations."""
    from analysis.portfolio_builder import PORTFOLIO_BASKETS

    lines = ["=== AVAILABLE PORTFOLIO BASKETS ===\n"]
    for b in PORTFOLIO_BASKETS:
        lines.append(
            f"  {b['name']}: Equity {b.get('equity_pct', 0)}% | "
            f"Debt {b.get('debt_pct', 0)}% | "
            f"Hybrid {b.get('hybrid_pct', 0)}% | "
            f"Risk: {b.get('risk_level', 'N/A')}"
        )
    return "\n".join(lines)


@function_tool
def get_brokerage_vs_performance_data(
    category: str = "",
    risk_profile: str = "moderate",
    top_n: int = 20,
) -> str:
    """Get funds plotted by AAUM (size), brokerage (revenue), and composite score.
    Identifies: high-AAUM + high-brokerage (stars), high-AAUM + low-brokerage (missed revenue),
    and high-brokerage + low-AAUM (growth potential).

    Args:
        category: Filter by category (Equity, Debt, Hybrid, etc). Empty = all.
        risk_profile: Risk profile: conservative, moderate, or aggressive.
        top_n: Number of funds per segment (default 20).
    """
    df = _load_ranked()
    df = df[df["risk_profile"] == risk_profile].copy()
    if category:
        df = df[df["category"].str.lower() == category.lower()]

    df = df.dropna(subset=["trail_brokerage_incl_gst", "aum_cr", "composite_score"])
    top_n = min(top_n, 25)

    cols = ["scheme_name", "amc", "sub_category", "composite_score",
            "trail_brokerage_incl_gst", "aum_cr", "return_1y_regular", "tieup_category"]

    # Stars: high AAUM + high brokerage
    aum_q75 = df["aum_cr"].quantile(0.75)
    brok_q75 = df["trail_brokerage_incl_gst"].quantile(0.75)
    brok_q25 = df["trail_brokerage_incl_gst"].quantile(0.25)
    stars = df[(df["aum_cr"] >= aum_q75) & (df["trail_brokerage_incl_gst"] >= brok_q75)]
    stars = stars.sort_values("aum_cr", ascending=False).head(top_n)

    # Missed revenue: high AAUM + low brokerage
    missed = df[(df["aum_cr"] >= aum_q75) & (df["trail_brokerage_incl_gst"] <= brok_q25)]
    missed = missed.sort_values("aum_cr", ascending=False).head(top_n)

    # Growth potential: high brokerage + low AAUM
    aum_q25 = df["aum_cr"].quantile(0.25)
    growth = df[(df["trail_brokerage_incl_gst"] >= brok_q75) & (df["aum_cr"] <= aum_q25)]
    growth = growth.sort_values("trail_brokerage_incl_gst", ascending=False).head(top_n)

    return (
        f"=== BROKERAGE vs PERFORMANCE ({len(df)} funds) ===\n"
        f"Avg AAUM: {df['aum_cr'].mean():.1f} Cr | Avg Brokerage: {df['trail_brokerage_incl_gst'].mean():.2f}%\n\n"
        f"STARS (High AAUM + High Brokerage) — {len(stars)} funds:\n"
        f"{stars[cols].to_string(index=False)}\n\n"
        f"MISSED REVENUE (High AAUM + Low Brokerage) — {len(missed)} funds:\n"
        f"{missed[cols].to_string(index=False)}\n\n"
        f"GROWTH POTENTIAL (High Brokerage + Low AAUM) — {len(growth)} funds:\n"
        f"{growth[cols].to_string(index=False)}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EMAIL SUMMARY TOOL
# ═══════════════════════════════════════════════════════════════════════════════

@function_tool
def send_email_summary(risk_profile: str = "moderate", recipient_name: str = "") -> str:
    """Generate and send the daily email summary with portfolio and client insights.
    Uses the email settings configured in the Email Summary page (SMTP, recipients).
    The user must have configured email settings before this can work.

    Args:
        risk_profile: Risk profile for analysis: conservative, moderate, or aggressive.
        recipient_name: Name for the greeting (optional, uses config default if empty).
    """
    from email_summary.sender import load_config, send_email
    from email_summary.generator import generate_email_html

    config = load_config()

    if not config.get("sender_email") or not config.get("sender_password"):
        return "Email not configured. Please set up SMTP settings in the Email Summary page first."
    if not config.get("recipients"):
        return "No recipients configured. Please add recipients in the Email Summary page first."

    name = recipient_name or config.get("recipient_name", "Manager")

    # Get portfolio data
    portfolio_df = _get_portfolio()

    # Get client data
    business_df = _client_data.get("business_df")
    gaps = _client_data.get("gaps")
    metrics = _client_data.get("metrics")

    html = generate_email_html(
        portfolio_df=portfolio_df,
        business_df=business_df,
        gaps=gaps,
        metrics=metrics,
        risk_profile=risk_profile,
        recipient_name=name,
    )

    result = send_email(html, config)

    if result == "ok":
        recipients = ", ".join(config["recipients"])
        return f"Email sent successfully to {recipients}."
    else:
        return f"Failed to send email: {result}"


# ═══════════════════════════════════════════════════════════════════════════════
# WEEKLY BEST FUND CARD TOOL
# ═══════════════════════════════════════════════════════════════════════════════

@function_tool
def generate_weekly_fund_card(
    risk_profile: str = "moderate",
    category: str = "",
    sub_category: str = "",
    pick_index: int = 0,
) -> str:
    """Generate a shareable 'Weekly Best Fund' image card. Returns the file path
    of the saved PNG image. The user can pick the Nth best fund using pick_index.

    Args:
        risk_profile: Risk profile: conservative, moderate, or aggressive.
        category: Filter by category (Equity, Debt, Hybrid, etc). Empty = overall best.
        sub_category: Filter by sub-category. Empty = all sub-categories.
        pick_index: 0 = best fund, 1 = second best, 2 = third best, etc.
    """
    from image_generator.weekly_card import get_weekly_best_funds, generate_card

    funds = get_weekly_best_funds(
        risk_profile=risk_profile,
        category=category,
        sub_category=sub_category,
        top_n=10,
    )

    if funds.empty:
        return "No funds found for the given filters."

    if pick_index >= len(funds):
        return f"Only {len(funds)} candidates available. Use pick_index 0 to {len(funds) - 1}."

    fund = funds.iloc[pick_index]
    img = generate_card(fund)

    out_dir = os.path.join(DATA_DIR, "..", "..", "generated_cards")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"weekly_best_{fund['scheme_name'].replace(' ', '_')[:30]}.png"
    out_path = os.path.join(out_dir, fname)
    img.save(out_path, format="PNG")

    return (
        f"Card generated for: {fund['scheme_name']}\n"
        f"  Score: {fund.get('composite_score', 'N/A')}, "
        f"1Y Return: {fund.get('return_1y_regular', 'N/A')}%, "
        f"Brokerage: {fund.get('trail_brokerage_incl_gst', 'N/A')}%\n"
        f"  Saved to: {out_path}\n"
        f"  (#{pick_index + 1} of {len(funds)} candidates)\n"
        f"  To get the next best, use pick_index={pick_index + 1}."
    )


@function_tool
def send_custom_email(subject: str, body: str) -> str:
    """Send a custom email with any content. Use this when the user asks to email
    a chat summary, a specific answer, analysis results, or any text.
    Uses the email settings configured in the Email Summary page.

    Args:
        subject: Email subject line.
        body: The email body content (plain text). Will be formatted into a styled HTML email.
    """
    from email_summary.sender import load_config
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from datetime import datetime

    config = load_config()
    if not config.get("sender_email") or not config.get("sender_password"):
        return "Email not configured. Please set up SMTP settings in the Email Summary page first."
    if not config.get("recipients"):
        return "No recipients configured. Please add recipients in the Email Summary page first."

    # Convert body to styled HTML
    body_html = body.replace("\n", "<br>")
    now = datetime.now().strftime("%d %B %Y, %I:%M %p")
    html = f"""
    <html><body style="font-family:'Segoe UI',Arial,sans-serif;background:#f8fafc;margin:0;padding:20px;">
    <div style="max-width:700px;margin:0 auto;background:white;border-radius:12px;
                box-shadow:0 1px 3px rgba(0,0,0,0.1);overflow:hidden;">
        <div style="background:linear-gradient(135deg,#3b82f6,#60a5fa);padding:24px 32px;color:white;">
            <h1 style="margin:0;font-size:20px;">{subject}</h1>
            <p style="margin:6px 0 0;opacity:0.85;font-size:13px;">Enkay Investments | {now}</p>
        </div>
        <div style="padding:24px 32px;font-size:14px;line-height:1.7;color:#1e293b;">
            {body_html}
        </div>
        <div style="background:#f8fafc;padding:12px 32px;text-align:center;color:#94a3b8;
                    font-size:11px;border-top:1px solid #e2e8f0;">
            Sent from Enkay Investments Fund Analytics
        </div>
    </div>
    </body></html>
    """

    sender = config["sender_email"]
    password = config["sender_password"]
    recipients = config["recipients"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(config.get("smtp_server", "smtp.gmail.com"),
                          config.get("smtp_port", 587)) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
        return f"Email sent successfully to {', '.join(recipients)}."
    except Exception as e:
        return f"Failed to send email: {e}"


ALL_TOOLS = [
    # Fund analysis tools (Fund Ranker page)
    get_top_funds,
    get_fund_details,
    get_category_summary,
    search_funds,
    compare_funds,
    # Peer Comparison page
    get_peer_comparison,
    # Fund Shift Advisor page
    get_fund_shift_alternatives,
    # Portfolio Exposure Review page
    review_portfolio_exposure,
    quantify_portfolio_exposure,
    get_portfolio_alternatives,
    # AMC Concentration page
    get_amc_analysis,
    get_amc_concentration_analysis,
    get_holdings_amc_concentration,
    # Brokerage vs Performance page
    get_brokerage_revenue_analysis,
    get_brokerage_vs_performance_data,
    get_high_potential_opportunities,
    # Recommended Portfolios page
    build_recommended_portfolio,
    list_portfolio_baskets,
    # Portfolio holdings tools
    get_portfolio_holdings,
    get_portfolio_concentration,
    # Client insights tools
    get_client_overview,
    get_top_clients,
    get_client_gaps,
    get_pareto_analysis,
    get_sip_leakage_analysis,
    get_client_growth_segments,
    # Email
    send_email_summary,
    send_custom_email,
    # Weekly Best Fund card
    generate_weekly_fund_card,
]
