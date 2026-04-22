"""
weekly_card.py
Generates a professional "Weekly Best Fund" shareable image card.
Design is fixed — only data variables change.
Output: 1080x1920 (9:16 phone aspect ratio)
"""
import os
import sys
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

from src.ai_copy import generate_weekly_caption

_SRC_DIR = os.path.join(os.path.dirname(__file__), "..")
for _p in [_SRC_DIR, os.path.join(_SRC_DIR, "analysis"), os.path.join(_SRC_DIR, "scoring")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
RANKED_FILE = os.path.join(BASE_DIR, "data", "processed", "ranked_funds.csv")
LOGO_FILE = os.path.join(BASE_DIR, "logo-2.png")

# ── Card width (height is calculated dynamically to fit content) ──
W = 1080
H = 1920

# ── Colors ──
BG = "#FFFFFF"
BLUE = "#3b82f6"
DARK_BLUE = "#1e3a5f"
NAVY = "#0f172a"
LIGHT_BG = "#f1f5f9"
GREEN = "#10b981"
RED = "#ef4444"
AMBER = "#f59e0b"
GRAY = "#64748b"
LIGHT_GRAY = "#e2e8f0"
ENKAY_BLUE = "#1a237e"

# ── Fonts ──
_FONT_CACHE = {}
_FONT_FALLBACK_WARNED = False


def _font(size, bold=False):
    global _FONT_FALLBACK_WARNED
    key = (size, bold)
    if key not in _FONT_CACHE:
        font_dir = os.path.join(os.path.dirname(__file__), "fonts")
        regular_paths = [
            os.path.join(font_dir, "LiberationSans-Regular.ttf"),
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        bold_paths = [
            os.path.join(font_dir, "LiberationSans-Bold.ttf"),
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\calibrib.ttf",
            "/System/Library/Fonts/ArialHB.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        paths = bold_paths if bold else regular_paths
        for p in paths:
            if os.path.exists(p):
                try:
                    if p.endswith(".ttc"):
                        idx = 1 if bold else 0
                        _FONT_CACHE[key] = ImageFont.truetype(p, size, index=idx)
                    else:
                        _FONT_CACHE[key] = ImageFont.truetype(p, size)
                    return _FONT_CACHE[key]
                except Exception:
                    continue
        if not _FONT_FALLBACK_WARNED:
            print("[WARN] Using PIL default font. Bundle TTF files under src/image_generator/fonts/ for consistent deployment.")
            _FONT_FALLBACK_WARNED = True
        _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def _round_rect(draw, xy, radius, fill):
    draw.rounded_rectangle(xy, radius=radius, fill=fill)


def _text_center(draw, y, text, font, fill):
    """Center text horizontally at y position (y is top-left of text). Use _text_center_vertical for vertical centering."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = round((W - tw) / 2)  # Round to nearest integer for crisp rendering
    draw.text((x, y), text, font=font, fill=fill)


def _text_center_vertical(draw, y, text, font, fill):
    """Center text horizontally AND vertically around y coordinate."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = round((W - tw) / 2)
    # Adjust y to center text vertically (y parameter is treated as the center point)
    adjusted_y = round(y - (th / 2))
    draw.text((x, adjusted_y), text, font=font, fill=fill)


def _format_return(val):
    if pd.isna(val):
        return "N/A"
    return f"{val:+.2f}%" if val != 0 else "0.00%"


def _return_color(val):
    if pd.isna(val):
        return GRAY
    return GREEN if val > 0 else RED if val < 0 else NAVY


def _load_logo():
    if os.path.exists(LOGO_FILE):
        return Image.open(LOGO_FILE).convert("RGBA")
    return None


def get_weekly_best_funds(risk_profile="moderate", category="", sub_category="", top_n=5):
    """Get top N funds for the week, ranked by composite score."""
    df = pd.read_csv(RANKED_FILE)
    df = df[df["risk_profile"] == risk_profile].copy()
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    if sub_category:
        df = df[df["sub_category"].str.lower() == sub_category.lower()]
    df = df.dropna(subset=["composite_score"])
    df = df.sort_values("composite_score", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def generate_card(fund_row: pd.Series) -> Image.Image:
    """Generate a shareable image card for a single fund in 1080x1920."""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    now = datetime.now()

    # ════════════════════════════════════════════════════
    # HEADER (dark blue, with logo)
    # ════════════════════════════════════════════════════
    header_h = 340
    _round_rect(draw, (0, 0, W, header_h), radius=0, fill=DARK_BLUE)
    draw.rectangle((0, header_h - 6, W, header_h), fill=BLUE)

    # Logo
    logo = _load_logo()
    if logo:
        logo_h = 70
        aspect = logo.width / logo.height
        # Clamp logo width to prevent extreme aspect ratios from breaking layout
        max_logo_w = 400  # ~37% of card width for safety
        logo_w = int(min(logo_h * aspect, max_logo_w))
        if logo_w != int(logo_h * aspect):
            # If clamped, recalculate height to maintain aspect
            logo_h = int(logo_w / aspect)
        logo_resized = logo.resize((logo_w, logo_h), Image.LANCZOS)
        logo_x = round((W - logo_w) / 2)  # Round for precise centering
        img.paste(logo_resized, (logo_x, 35), logo_resized)
        title_y = 120
    else:
        _text_center(draw, 50, "Enkay Investments", _font(42, bold=True), "#FFFFFF")
        title_y = 110

    # "WEEKLY BEST FUND" title
    _text_center(draw, title_y, "WEEKLY BEST FUND", _font(28, bold=True), "#FFFFFF")

    # Week info
    week_str = f"Week of {now.strftime('%d %B %Y')}"
    _text_center(draw, title_y + 45, week_str, _font(20), "#94a3b8")

    # Category badge
    category = str(fund_row.get("category", ""))
    sub_category = str(fund_row.get("sub_category", ""))
    badge_text = f"{category}  |  {sub_category}" if category and sub_category else category or sub_category
    if badge_text:
        bbox = draw.textbbox((0, 0), badge_text, font=_font(18))
        bw = bbox[2] - bbox[0] + 40
        bx = round((W - bw) / 2)
        _round_rect(draw, (bx, title_y + 90, bx + bw, title_y + 122), radius=16, fill=BLUE)
        _text_center(draw, title_y + 94, badge_text, _font(18), "#FFFFFF")

    # Rank badge (top right)
    rank = fund_row.get("rank", "")
    if pd.notna(rank):
        rank_text = f"#{int(rank)}"
        _round_rect(draw, (W - 130, header_h - 65, W - 30, header_h - 15), radius=10, fill=GREEN)
        bbox = draw.textbbox((0, 0), rank_text, font=_font(30, bold=True))
        tw = bbox[2] - bbox[0]
        draw.text((round(W - 80 - tw / 2), header_h - 58), rank_text, font=_font(30, bold=True), fill="#FFFFFF")
        draw.text((40, header_h - 50), "RANK", font=_font(16), fill="#94a3b8")

    # ════════════════════════════════════════════════════
    # FUND NAME SECTION
    # ════════════════════════════════════════════════════
    fund_name = str(fund_row.get("scheme_name", "Unknown Fund"))
    amc = str(fund_row.get("amc", ""))

    name_font = _font(38, bold=True)
    words = fund_name.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=name_font)
        if bbox[2] - bbox[0] > W - 120:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)

    y = header_h + 40
    for line in lines[:3]:
        _text_center(draw, y, line, name_font, NAVY)
        y += 50

    if amc:
        _text_center(draw, y + 5, amc, _font(22), GRAY)
        y += 40

    caption = generate_weekly_caption(fund_row.to_dict())
    if caption:
        caption_font = _font(20)
        bbox = draw.textbbox((0, 0), caption, font=caption_font)
        while caption and (bbox[2] - bbox[0]) > W - 160:
            caption = " ".join(caption.split()[:-1])
            bbox = draw.textbbox((0, 0), caption, font=caption_font)
        _text_center(draw, y + 5, caption, caption_font, BLUE)
        y += 34

    # Separator
    y += 15
    draw.line((80, y, W - 80, y), fill=LIGHT_GRAY, width=2)
    y += 30

    # ════════════════════════════════════════════════════
    # RETURNS SECTION
    # ════════════════════════════════════════════════════
    _text_center(draw, y, "RETURNS", _font(18), GRAY)
    y += 40

    ret_1y = fund_row.get("return_1y_regular")
    ret_3y = fund_row.get("return_3y_regular")
    ret_5y = fund_row.get("return_5y_regular")

    box_w = 290
    box_h = 130
    gap = 30
    total_w = box_w * 3 + gap * 2
    start_x = round((W - total_w) / 2)  # Round for integer pixel alignment

    for i, (label, val) in enumerate([("1 Year", ret_1y), ("3 Year", ret_3y), ("5 Year", ret_5y)]):
        bx = int(start_x + i * (box_w + gap))  # Ensure integer pixel position
        _round_rect(draw, (bx, y, bx + box_w, y + box_h), radius=14, fill=LIGHT_BG)

        draw.text((bx + 22, y + 15), label, font=_font(18), fill=GRAY)

        val_text = _format_return(val)
        val_color = _return_color(val)
        val_font = _font(34, bold=True)
        bbox = draw.textbbox((0, 0), val_text, font=val_font)
        vw = bbox[2] - bbox[0]
        draw.text((round(bx + (box_w - vw) / 2), y + 60), val_text, font=val_font, fill=val_color)

    y += box_h + 35

    # ════════════════════════════════════════════════════
    # KEY METRICS GRID
    # ════════════════════════════════════════════════════
    _text_center(draw, y, "KEY METRICS", _font(18), GRAY)
    y += 40

    score = fund_row.get("composite_score")
    aum = fund_row.get("aum_cr")
    brokerage = fund_row.get("trail_brokerage_incl_gst")
    riskometer = fund_row.get("riskometer", "")
    tieup = fund_row.get("tieup_category", "")

    metrics = [
        ("Composite Score", f"{score:.1f} / 100" if pd.notna(score) else "N/A", BLUE),
        ("AAUM", f"{aum:,.0f} Cr" if pd.notna(aum) else "N/A", NAVY),
        ("Trail Brokerage", f"{brokerage:.2f}%" if pd.notna(brokerage) else "N/A", GREEN),
        ("TieUp Category", str(tieup) if pd.notna(tieup) and tieup else "None", AMBER),
    ]

    m_box_w = (W - 120 - 24) // 2
    m_box_h = 100
    for i, (label, val_text, color) in enumerate(metrics):
        col = i % 2
        row = i // 2
        mx = 60 + col * (m_box_w + 24)
        my = y + row * (m_box_h + 16)

        _round_rect(draw, (mx, my, mx + m_box_w, my + m_box_h), radius=12, fill=LIGHT_BG)
        draw.rounded_rectangle((mx, my, mx + 6, my + m_box_h), radius=3, fill=color)

        draw.text((mx + 24, my + 15), label, font=_font(16), fill=GRAY)
        draw.text((mx + 24, my + 45), val_text, font=_font(30, bold=True), fill=NAVY)

    y += 2 * (m_box_h + 16) + 30

    # ════════════════════════════════════════════════════
    # RISK LEVEL BADGE
    # ════════════════════════════════════════════════════
    if riskometer and pd.notna(riskometer):
        risk_colors = {
            "Low": GREEN, "Moderately Low": "#22c55e",
            "Moderate": AMBER, "Moderately High": "#f97316",
            "High": "#ef4444", "Very High": "#dc2626",
        }
        r_color = risk_colors.get(str(riskometer), GRAY)
        r_text = f"Risk: {riskometer}"
        bbox = draw.textbbox((0, 0), r_text, font=_font(20))
        rw = bbox[2] - bbox[0] + 40
        rx = round((W - rw) / 2)
        _round_rect(draw, (rx, y, rx + rw, y + 42), radius=21, fill=r_color)
        _text_center(draw, y + 9, r_text, _font(20, bold=True), "#FFFFFF")
        y += 60

    # ════════════════════════════════════════════════════
    # BENCHMARK INFO (if available)
    # ════════════════════════════════════════════════════
    benchmark = fund_row.get("benchmark", "")
    if benchmark and pd.notna(benchmark):
        _text_center(draw, y + 10, f"Benchmark: {benchmark}", _font(16), GRAY)
        y += 45

    # ════════════════════════════════════════════════════
    # FOOTER (positioned right after content)
    # ════════════════════════════════════════════════════
    y += 30  # padding before footer
    footer_h = 110
    footer_y = H - footer_h
    y = min(y, footer_y - 80)
    final_h = H

    draw.rectangle((0, footer_y, W, final_h), fill=DARK_BLUE)
    draw.rectangle((0, footer_y, W, footer_y + 4), fill=BLUE)

    # Logo in footer (smaller)
    if logo:
        f_logo_h = 35
        f_aspect = logo.width / logo.height
        f_logo_w = int(f_logo_h * f_aspect)
        f_logo_resized = logo.resize((f_logo_w, f_logo_h), Image.LANCZOS)
        f_logo_x = round((W - f_logo_w) / 2)  # Round for consistent centering
        img.paste(f_logo_resized, (f_logo_x, footer_y + 15), f_logo_resized)

    _text_center(draw, footer_y + 58,
                 "Mutual fund investments are subject to market risks.",
                 _font(13), "#64748b")
    _text_center(draw, footer_y + 76,
                 "Read all scheme related documents carefully.",
                 _font(13), "#64748b")
    _text_center(draw, footer_y + 93,
                 f"Generated on {now.strftime('%d %b %Y')}",
                 _font(12), "#475569")
    return img
