"""
amc_concentration.py
Computes AMC-level exposure among top-ranked funds.
Flags any AMC with > 30% share in the selected top-N funds.
Suggests rebalancing candidates from under-represented AMCs.
Also computes AMC concentration from current AUM holdings.
Used by the dashboard AMC Concentration tab.
"""
import os
import pandas as pd
from rapidfuzz import process, fuzz

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..", "..")
RANKED_FILE = os.path.join(BASE_DIR, "data", "processed", "ranked_funds.csv")
AUM_FILE    = os.path.join(BASE_DIR, "Scheme_wise_AUM.xls")

CONCENTRATION_ALERT_THRESHOLD = 0.30   # 30%
FUZZY_THRESHOLD = 75


def get_top_funds(
    df: pd.DataFrame,
    risk_profile: str = "moderate",
    top_n_per_subcat: int = 5,
    categories: list = None,
) -> pd.DataFrame:
    """Return top-N funds per sub-category filtered by risk profile (and optional category list)."""
    subset = df[df["risk_profile"] == risk_profile]
    if categories:
        subset = subset[subset["category"].isin(categories)]
    return (
        subset
        .sort_values("rank")
        .groupby("sub_category", group_keys=False)
        .head(top_n_per_subcat)
        .reset_index(drop=True)
    )


def compute_amc_concentration(
    risk_profile: str = "moderate",
    top_n_per_subcat: int = 5,
    categories: list = None,
    df: pd.DataFrame = None,
) -> dict:
    """
    Returns a dict with:
      - 'summary'  : DataFrame of AMC → count, pct, tieup_category, alert
      - 'top_funds': DataFrame of the top funds used
      - 'rebalance': DataFrame of rebalancing suggestions (funds from low-exposure AMCs)
      - 'alerts'   : list of AMC names that exceed CONCENTRATION_ALERT_THRESHOLD
    """
    if df is None:
        df = pd.read_csv(RANKED_FILE)

    top_funds = get_top_funds(df, risk_profile, top_n_per_subcat, categories)

    # ── AMC exposure summary ──────────────────────────────────────────────
    total = len(top_funds)
    amc_counts = (
        top_funds.groupby("amc")
        .agg(
            count         = ("scheme_name", "count"),
            avg_composite  = ("composite_score", "mean"),
            tieup_category = ("tieup_category", "first"),
        )
        .reset_index()
    )
    amc_counts["pct"]   = (amc_counts["count"] / total).round(4)
    amc_counts["alert"] = amc_counts["pct"] > CONCENTRATION_ALERT_THRESHOLD
    amc_counts = amc_counts.sort_values("pct", ascending=False).reset_index(drop=True)

    alerts = amc_counts[amc_counts["alert"]]["amc"].tolist()

    # ── Rebalancing suggestions ──────────────────────────────────────────
    # Find AMCs with low representation that still have high-scoring funds
    low_amc = set(df[df["risk_profile"] == risk_profile]["amc"].unique()) - set(
        amc_counts[amc_counts["count"] >= 2]["amc"].tolist()
    )
    all_profile = df[df["risk_profile"] == risk_profile]
    if categories:
        all_profile = all_profile[all_profile["category"].isin(categories)]

    rebalance = (
        all_profile[all_profile["amc"].isin(low_amc)]
        .sort_values("composite_score", ascending=False)
        .groupby(["amc", "sub_category"])
        .head(1)
        .nlargest(10, "composite_score")
        [["scheme_name", "amc", "sub_category", "tieup_category",
          "composite_score", "trail_brokerage_incl_gst", "rank"]]
        .reset_index(drop=True)
    )

    return {
        "summary":   amc_counts,
        "top_funds": top_funds,
        "rebalance": rebalance,
        "alerts":    alerts,
        "total_funds_analysed": total,
    }


def load_aum_data(uploaded_file=None):
    """Load and clean AUM data from uploaded file or default Scheme_wise_AUM.xls."""
    import io

    if uploaded_file is not None:
        if hasattr(uploaded_file, "getvalue"):
            file_bytes = uploaded_file.getvalue()
        else:
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
            file_bytes = uploaded_file.read()
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        sheet_names_to_try = ["AUM Report", "AUM", "Sheet1", "Sheet0", xl.sheet_names[0]]
        df = None

        for sheet_name in sheet_names_to_try:
            try:
                temp_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=1)
                if any("scheme" in str(c).lower() for c in temp_df.columns):
                    df = temp_df
                    break
            except Exception:
                continue

        if df is None:
            df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)
    else:
        if not os.path.exists(AUM_FILE):
            raise ValueError(
                "No default Scheme_wise_AUM.xls found. Please upload the Scheme-wise AAUM file (.xls/.xlsx)."
            )
        df = pd.read_excel(AUM_FILE, sheet_name="AUM Report", header=1)

    df.columns = ["sr_no", "amc", "scheme", "nav", "equity", "debt", "hybrid", "physical_assets", "others", "total"]
    df = df.dropna(subset=["scheme"])
    df["scheme"] = df["scheme"].astype(str).str.strip()
    df["amc"] = df["amc"].astype(str).str.strip()
    df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0)
    return df


def fuzzy_match_amc(aum_amc, ranked_amcs, threshold=70):
    """Match AMC names from AUM to ranked funds using fuzzy matching."""
    for rank_amc in ranked_amcs:
        score = fuzz.token_sort_ratio(aum_amc.lower(), rank_amc.lower())
        if score >= threshold:
            return rank_amc
    return aum_amc


def compute_current_amc_concentration(
    aum_threshold=2500000,
    ranked_df=None,
    uploaded_file=None,
) -> dict:
    """
    Compute AMC concentration from current AUM holdings.
    
    Returns:
      dict with:
        - 'summary': DataFrame of AMC -> aum, pct, alert
        - 'total_aum': total AUM amount
        - 'total_amcs': number of unique AMCs
        - 'schemes_matched': number of schemes matched to ranked funds
    """
    aum_df = load_aum_data(uploaded_file=uploaded_file)
    return compute_current_amc_concentration_from_df(
        aum_df,
        aum_threshold=aum_threshold,
        ranked_df=ranked_df,
    )


def compute_current_amc_concentration_from_df(
    aum_df: pd.DataFrame,
    aum_threshold=2500000,
    ranked_df=None,
) -> dict:
    """Compute AMC concentration from an in-memory AUM DataFrame."""
    if aum_df is None or aum_df.empty:
        empty_summary = pd.DataFrame(columns=["amc", "aum", "pct", "alert"])
        return {
            "summary": empty_summary,
            "total_aum": 0.0,
            "total_amcs": 0,
            "schemes_matched": 0,
        }
    
    aum_df = aum_df[aum_df["total"] >= aum_threshold].copy()

    if aum_df.empty:
        empty_summary = pd.DataFrame(columns=["amc", "aum", "pct", "alert"])
        return {
            "summary": empty_summary,
            "total_aum": 0.0,
            "total_amcs": 0,
            "schemes_matched": 0,
        }
    
    aum_by_amc = aum_df.groupby("amc")["total"].sum().reset_index()
    total_aum = aum_by_amc["total"].sum()

    if total_aum <= 0:
        empty_summary = pd.DataFrame(columns=["amc", "aum", "pct", "alert"])
        return {
            "summary": empty_summary,
            "total_aum": 0.0,
            "total_amcs": 0,
            "schemes_matched": len(aum_df),
        }
    
    aum_by_amc["pct"] = (aum_by_amc["total"] / total_aum).round(4)
    aum_by_amc["alert"] = aum_by_amc["pct"] > CONCENTRATION_ALERT_THRESHOLD
    aum_by_amc = aum_by_amc.sort_values("total", ascending=False).reset_index(drop=True)
    aum_by_amc.columns = ["amc", "aum", "pct", "alert"]
    
    schemes_matched = len(aum_df)
    
    return {
        "summary": aum_by_amc,
        "total_aum": total_aum,
        "total_amcs": len(aum_by_amc),
        "schemes_matched": schemes_matched,
    }


if __name__ == "__main__":
    result = compute_amc_concentration(risk_profile="moderate", top_n_per_subcat=5)
    print("=== AMC Concentration (Moderate, Top-5 per sub-category) ===")
    print(result["summary"].to_string(index=False))
    if result["alerts"]:
        print(f"\n⚠ CONCENTRATION ALERT: {result['alerts']}")
    print(f"\nRebalancing suggestions:\n{result['rebalance'].to_string(index=False)}")
