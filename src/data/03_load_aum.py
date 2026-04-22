"""
03_load_aum.py
Parses the AMFI Average AUM Excel file (average-aum.xlsx).
The file has a hierarchical structure:
  - Row 0 (header): Quarter label
  - Data row 0: column headers (AMFI Code, Scheme NAV Name, AUM excl FoF, FoF AUM)
  - AMC name rows: section headers for each mutual fund house
  - Category rows: sub-section headers (e.g., "Equity Scheme - Large Cap Fund")
  - Data rows: numeric AMFI Code, Scheme Name, AUM values (in Lakhs)
  - "Total" rows: aggregation rows to skip

Logic:
  1. Include ALL plan-options (Direct + Regular, Growth + IDCW + Bonus etc.)
  2. Strip plan/option suffixes to get base fund name
  3. Sum AAUM across ALL variants for each scheme
  4. Convert from Lakhs to Crores (÷ 100)
  5. Output column: aum_cr (Average AUM for the quarter)

Outputs: data/processed/scheme_aum.csv
"""
import os
import re
import pandas as pd
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.join(os.path.dirname(__file__), "..", "..")
IN_FILE   = os.path.join(BASE_DIR, "average-aum.xlsx")
OUT_PATH  = os.path.join(BASE_DIR, "data", "processed", "scheme_aum.csv")

# Dash pattern: matches " - ", "-", " – ", "–" with optional surrounding spaces
_DASH = r'\s*[-–]\s*'

# Frequency keywords that may precede IDCW/Dividend (e.g. "Daily IDCW", "Monthly Dividend")
_FREQ = r'(?:Daily|Weekly|Fortnightly|Monthly|Quarterly|Half\s+Yearly|Annual)\s+'

# Patterns to strip from the scheme name to get the base fund name.
# Applied in order — first match wins, so longer/more specific patterns first.
PAYOUT_SUFFIXES = [
    # "(Direct|Regular) Plan" followed by anything
    re.compile(_DASH + r'(Direct|Regular)\s+Plan\b.*$', re.IGNORECASE),
    re.compile(r'\s+(Direct|Regular)\s+Plan\b.*$', re.IGNORECASE),
    # "(Direct|Regular)" WITHOUT "Plan", followed by anything (Growth/IDCW/Income Distribution/...)
    # Matches: "FUND DIRECT GROWTH", "Fund-Direct - IDCW", "Fund - Regular Fortnightly IDCW"
    # The separator after Direct/Regular can be space, dash, or space-dash-space.
    re.compile(_DASH + r'(Direct|Regular)[\s\-–]+.*$', re.IGNORECASE),
    re.compile(r'\s+(Direct|Regular)[\s\-–]+.*$', re.IGNORECASE),
    # "Retail Plan" or "Retail Option" followed by anything
    re.compile(_DASH + r'Retail\s+(Plan|Option)\b.*$', re.IGNORECASE),
    re.compile(_DASH + r'Retail\s*$', re.IGNORECASE),
    # "(Growth|IDCW|Dividend|Bonus) Plan" followed by anything
    re.compile(_DASH + r'(Growth|IDCW|Dividend|Bonus)\s+Plan\b.*$', re.IGNORECASE),
    re.compile(r'\s+(Growth|IDCW|Dividend|Bonus)\s+Plan\b.*$', re.IGNORECASE),
    # Frequency + IDCW/Dividend + Option: "- Daily IDCW Option", "- Monthly Dividend Payout"
    re.compile(_DASH + _FREQ + r'(IDCW|Dividend|Bonus)\s*(Option|Payout|Reinvestment)?\b.*$', re.IGNORECASE),
    # Plain "- IDCW/Dividend/Bonus Option/Payout", "- Growth Option"
    re.compile(_DASH + r'(Growth|IDCW|Dividend|Bonus)\s+(Option|Payout|Reinvestment)\b.*$', re.IGNORECASE),
    # Trailing frequency + IDCW without Option: "- Daily IDCW", "- Weekly IDCW"
    re.compile(_DASH + _FREQ + r'(IDCW|Dividend|Bonus)\s*$', re.IGNORECASE),
    # Trailing "- Growth", "- IDCW", "- Dividend", "- Bonus", "- Regular", "- Direct"
    re.compile(_DASH + r'(Regular|Direct|Growth|IDCW|Dividend|Bonus)\s*$', re.IGNORECASE),
]


def extract_base_fund_name(scheme_name: str) -> str:
    """
    Strip the plan type and payout option suffix to get the base fund name.
    Handles both Direct and Regular plans so they group together.

    Examples:
      'Kotak Arbitrage Fund - Direct Plan - Growth'          → 'Kotak Arbitrage Fund'
      'Kotak Arbitrage Fund - Regular Plan - IDCW - Payout'  → 'Kotak Arbitrage Fund'
      'KOTAK FLOATING RATE FUND-DIRECT PLAN-GROWTH OPTION'   → 'KOTAK FLOATING RATE FUND'
      'KOTAK FLOATING RATE FUND-REGULAR PLAN-STANDARD ...'   → 'KOTAK FLOATING RATE FUND'
    """
    if not isinstance(scheme_name, str):
        return str(scheme_name)

    cleaned = scheme_name.strip()
    for pattern in PAYOUT_SUFFIXES:
        cleaned = pattern.sub("", cleaned).strip()
    # Normalize multiple spaces to single space (e.g. "Kotak  Arbitrage" → "Kotak Arbitrage")
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned


def parse_aum_excel(filepath: str) -> pd.DataFrame:
    """
    Parse the AMFI AUM Excel file with its hierarchical structure.
    - Includes ALL plan-options (Direct + Regular, Growth + IDCW + Bonus etc.)
    - Strips plan/option suffixes to get base fund name
    - Sums AAUM across ALL variants for each scheme
    - Converts from Lakhs to Crores
    Returns a DataFrame with columns: scheme_code, scheme_name, aum_cr
    """
    # Read raw — the file has a single header row followed by mixed data
    raw = pd.read_excel(filepath, engine="openpyxl", header=None)

    # Col 0: AMFI Code (or AMC name / category name / "Total")
    # Col 1: Scheme NAV Name
    # Col 2: AUM excl FoF (Rs in Lakhs)
    # Col 3: FoF AUM (Rs in Lakhs)
    raw.columns = ["col0", "col1", "col2", "col3"]

    # Skip the first two metadata rows (quarter title + column headers)
    raw = raw.iloc[2:].reset_index(drop=True)

    # Identify data rows: those where col0 is a valid numeric AMFI code
    raw["col0_numeric"] = pd.to_numeric(raw["col0"], errors="coerce")
    data_rows = raw[raw["col0_numeric"].notna()].copy()

    # Keep ALL rows (Direct + Regular) — we sum everything per scheme
    data_rows["name_str"] = data_rows["col1"].astype(str)

    print(f"  Total data rows (all plans): {len(data_rows)}")

    # Parse AUM values (both columns contribute to total AUM)
    data_rows["aum_excl_fof"] = pd.to_numeric(data_rows["col2"], errors="coerce").fillna(0)
    data_rows["aum_fof"] = pd.to_numeric(data_rows["col3"], errors="coerce").fillna(0)
    data_rows["aum_lakhs"] = data_rows["aum_excl_fof"] + data_rows["aum_fof"]

    # Extract base fund name for grouping
    data_rows["base_fund_name"] = data_rows["name_str"].apply(extract_base_fund_name)

    # Case-insensitive grouping key (e.g. "SBI EQUITY HYBRID FUND" == "SBI Equity Hybrid Fund")
    data_rows["group_key"] = data_rows["base_fund_name"].str.lower()

    # Keep one scheme_code per fund (use the first/smallest code)
    data_rows["scheme_code"] = data_rows["col0_numeric"].astype(int)

    # Group by case-normalized key: sum AUM, keep first scheme_code and original name
    grouped = data_rows.groupby("group_key", as_index=False).agg(
        scheme_code=("scheme_code", "first"),
        base_fund_name=("base_fund_name", "first"),
        aum_lakhs=("aum_lakhs", "sum"),
        variant_count=("scheme_code", "count"),
    )

    # Convert Lakhs → Crores (÷ 100)
    grouped["aum_cr"] = (grouped["aum_lakhs"] / 100.0).round(2)

    result = pd.DataFrame({
        "scheme_code": grouped["scheme_code"],
        "scheme_name": grouped["base_fund_name"].str.strip(),
        "aum_cr": grouped["aum_cr"],
    })

    # Drop rows with zero or negative AUM
    result = result[result["aum_cr"] > 0].copy()

    multi_variant = grouped[grouped["variant_count"] > 1]
    print(f"  Funds with multiple plan-options aggregated: {len(multi_variant)}")
    print(f"  Total unique schemes (all plans summed): {len(result)}")

    return result.reset_index(drop=True)


def main(filepath: str = None):
    aum_file = filepath or IN_FILE

    if not os.path.exists(aum_file):
        print(f"[WARN] AUM file not found: {aum_file}")
        print("  Upload the latest average-aum.xlsx from AMFI website.")
        return

    print(f"Reading AUM file: {aum_file}")
    df = parse_aum_excel(aum_file)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\n[OK] scheme_aum.csv saved -> {OUT_PATH}")
    print(f"   Total unique schemes: {len(df)}")
    print(f"   AAUM range: {df['aum_cr'].min():.2f} Cr – {df['aum_cr'].max():.2f} Cr")
    print(f"   Median AAUM: {df['aum_cr'].median():.2f} Cr")
    print(f"   Total AAUM: {df['aum_cr'].sum():,.0f} Cr")


if __name__ == "__main__":
    main()
