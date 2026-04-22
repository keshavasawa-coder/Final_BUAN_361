"""Minimal-token AI copy helpers for summaries and fund captions."""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from functools import lru_cache
from typing import Any

from agents import Agent, Runner
from pydantic import BaseModel, Field


AI_COPY_MODEL = os.getenv("ENKAY_AI_COPY_MODEL", "gpt-4o-mini")
MAX_BULLETS = 4
MAX_BULLET_WORDS = 16
MAX_CAPTION_WORDS = 18


class BulletCopy(BaseModel):
    bullets: list[str] = Field(default_factory=list)


class CaptionCopy(BaseModel):
    caption: str = ""


def _strip_bullet_prefix(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[\s\-•*\d.\)]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).rstrip(".,;:!")


def _fallback_email_bullets(summary: dict[str, Any]) -> list[str]:
    bullets: list[str] = []
    portfolio = summary.get("portfolio", {})
    client = summary.get("client", {})
    over_names = portfolio.get("top_overexposed_names", [])
    under_names = portfolio.get("top_underutilized_names", [])
    high_aum_names = client.get("high_aum_no_sip_names", [])
    reduced_names = client.get("reduced_sip_names", [])
    terminated_names = client.get("sip_terminated_names", [])
    topup_names = client.get("no_topup_names", [])

    def _first_name(items: list[Any]) -> str:
        if not items:
            return ""
        return str(items[0]).strip()

    if portfolio.get("overexposed_count", 0):
        name = _first_name(over_names)
        bullets.append(
            f"Review {name} and other overexposed schemes before trimming weaker allocations."
            if name else
            "Review overexposed schemes and trim weaker allocations."
        )
    if portfolio.get("underutilized_count", 0):
        name = _first_name(under_names)
        bullets.append(
            f"Shift more capital toward {name} and similar high-quality underutilized funds."
            if name else
            "Shift more capital toward high-quality underutilized funds."
        )
    if client.get("high_aum_no_sip", 0):
        name = _first_name(high_aum_names)
        bullets.append(
            f"Call {name} and other high-AUM clients without SIPs to reopen recurring flows."
            if name else
            "Call high-AUM clients without SIPs and reopen recurring flows."
        )
    if client.get("reduced_sip", 0) or client.get("sip_terminated", 0):
        name = _first_name(reduced_names or terminated_names)
        bullets.append(
            f"Follow up on {name} before the reduced or stopped SIP momentum fades."
            if name else
            "Follow up on reduced or stopped SIPs before momentum fades."
        )
    if client.get("no_topup", 0):
        name = _first_name(topup_names)
        bullets.append(
            f"Push top-ups for {name} and other active SIP clients with room to invest more."
            if name else
            "Push top-ups for active SIP clients with room to invest more."
        )

    if not bullets:
        bullets.append("Review the latest portfolio summary and prioritize the weakest holdings.")
        bullets.append("Check client gaps and convert one-off investors into SIPs.")

    return bullets[:MAX_BULLETS]


def _fallback_caption(fund: dict[str, Any]) -> str:
    scheme_name = str(fund.get("scheme_name") or "this fund").strip()
    amc = str(fund.get("amc") or "").strip()
    rank = fund.get("rank")
    category = str(fund.get("category") or "").strip()
    sub_category = str(fund.get("sub_category") or "").strip()
    score = fund.get("composite_score")
    return_1y = fund.get("return_1y_regular")

    parts: list[str] = []
    if scheme_name:
        parts.append(scheme_name)
    if amc:
        parts.append(f"from {amc}")
    if rank is not None and not (isinstance(rank, float) and (rank != rank)):
        parts.append(f"rank #{int(rank)}")
    if category and sub_category:
        parts.append(f"in {category} / {sub_category}")
    elif category:
        parts.append(f"in {category}")
    if isinstance(score, (int, float)):
        parts.append(f"score {score:.0f}")
    if isinstance(return_1y, (int, float)):
        parts.append(f"1Y {return_1y:+.1f}%")

    if not parts:
        return _limit_words("A disciplined fund with compelling long-term consistency.", MAX_CAPTION_WORDS)

    sentence = "; ".join(parts[:3])
    if isinstance(score, (int, float)) and score >= 80:
        closing = "stands out for quality and consistency."
    elif isinstance(return_1y, (int, float)) and return_1y > 0:
        closing = "shows steady momentum and investable strength."
    else:
        closing = "offers a balanced, advisor-friendly profile."

    return _limit_words(f"{sentence} - {closing}", MAX_CAPTION_WORDS)


@lru_cache(maxsize=1)
def _email_agent() -> Agent:
    return Agent(
        name="Enkay Email To-Do Writer",
        instructions=(
            "Write concise advisor action bullets from the provided summary. "
            "Return JSON with a bullets array only. "
            "Use imperative voice, one short sentence per bullet, no markdown, no preamble, no numbering. "
            "Whenever the summary includes scheme or client names, mention at least one specific name in each bullet. "
            f"Provide at most {MAX_BULLETS} bullets and keep each bullet under {MAX_BULLET_WORDS} words. "
            "Do not mention AI, models, or uncertainty. Keep wording practical and compliance-safe."
        ),
        output_type=BulletCopy,
        model=AI_COPY_MODEL,
    )


@lru_cache(maxsize=1)
def _caption_agent() -> Agent:
    return Agent(
        name="Enkay Weekly Caption Writer",
        instructions=(
            "Write one short client-friendly caption for a fund card. "
            "Return JSON with a caption field only. "
            f"Keep it under {MAX_CAPTION_WORDS} words. "
            "Make it positive and persuasive, but avoid guarantees, exaggerated claims, or absolute language. "
            "Focus on quality, consistency, risk-adjusted appeal, or portfolio fit. No markdown, no preamble."
        ),
        output_type=CaptionCopy,
        model=AI_COPY_MODEL,
    )


def _run_blocking(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _runner():
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive fallback path
            error["exc"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if error:
        raise error["exc"]
    return result.get("value")


def _extract_bullets(output: BulletCopy | dict[str, Any] | None) -> list[str]:
    if output is None:
        return []
    if isinstance(output, BulletCopy):
        raw_bullets = output.bullets
    else:
        raw_bullets = output.get("bullets", [])

    cleaned: list[str] = []
    for bullet in raw_bullets:
        text = _limit_words(_strip_bullet_prefix(str(bullet)), MAX_BULLET_WORDS)
        if text:
            cleaned.append(text)
    return cleaned[:MAX_BULLETS]


def generate_email_bullets(summary: dict[str, Any]) -> list[str]:
    """Generate short action bullets for the daily email summary."""
    try:
        prompt = json.dumps(summary, separators=(",", ":"), ensure_ascii=False)
        result = _run_blocking(Runner.run(_email_agent(), prompt))
        bullets = _extract_bullets(getattr(result, "final_output", None))
        return bullets or _fallback_email_bullets(summary)
    except Exception:
        return _fallback_email_bullets(summary)


def generate_weekly_caption(fund: dict[str, Any]) -> str:
    """Generate a one-line caption for the weekly best fund card."""
    try:
        prompt = json.dumps(fund, separators=(",", ":"), ensure_ascii=False, default=str)
        result = _run_blocking(Runner.run(_caption_agent(), prompt))
        output = getattr(result, "final_output", None)
        caption = output.caption if isinstance(output, CaptionCopy) else str((output or {}).get("caption", ""))
        caption = _limit_words(_strip_bullet_prefix(caption), MAX_CAPTION_WORDS)
        return caption or _fallback_caption(fund)
    except Exception:
        return _fallback_caption(fund)