"""Minimal resume extraction via OpenAI-compatible chat completion."""

import json
import logging
import os
import re
from typing import Any

from schemas import EducationItem, ExperienceItem, ResumeExtraction

logger = logging.getLogger("resume_extract")

def _get_model_id() -> str:
    model = (os.environ.get("MODEL_ID") or "").strip()
    if model:
        return model
    return "qwen2.5-14b-instruct-1m"


_PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{7,}\d")


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        s = (item or "").strip()
        if not s:
            continue
        key = re.sub(r"\s+", " ", s)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _normalize_phone(value: str) -> str:
    s = (value or "").strip()
    s = re.sub(r"^(mob|mobile|tel|phone)\s*[:\-]\s*", "", s, flags=re.I)
    m = _PHONE_RE.search(s)
    s = m.group(0) if m else s
    return re.sub(r"\s+", " ", s).strip()


def _is_phone_like(value: str) -> bool:
    digits = re.sub(r"\D", "", value or "")
    return len(digits) >= 9


def _extract_json_candidates(text: str) -> list[str]:
    candidates = [text.strip()]
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.I)
    if fence:
        candidates.append(fence.group(1).strip())
    block = re.search(r"\{[\s\S]*\}", text)
    if block:
        candidates.append(block.group(0).strip())
    return candidates


def _normalize_model_payload(payload: dict[str, Any]) -> ResumeExtraction:
    """Lightweight normalization only — keep model output as-is as much as possible."""
    data: dict[str, Any] = dict(payload)

    # Some models wrap with {"result": {...}}
    if "result" in data and isinstance(data["result"], dict):
        data = dict(data["result"])

    # Keep only first email if multiple are joined in one string.
    email = data.get("email")
    if isinstance(email, str):
        emails = re.findall(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", email)
        if emails:
            data["email"] = emails[0]

    # Normalize and deduplicate phone numbers.
    phones: list[str] = []
    phone = data.get("phone")
    if isinstance(phone, str):
        p = _normalize_phone(phone)
        if _is_phone_like(p):
            phones.append(p)
            data["phone"] = p

    raw_phones = data.get("phones")
    if isinstance(raw_phones, list):
        for p in raw_phones:
            if isinstance(p, str):
                pn = _normalize_phone(p)
                if _is_phone_like(pn):
                    phones.append(pn)

    phones = _dedup_keep_order(phones)
    data["phones"] = phones
    if phones and not data.get("phone"):
        data["phone"] = phones[0]

    # Flatten description if model returned it as a list.
    exp = data.get("experience")
    if isinstance(exp, list):
        fixed: list[dict[str, Any]] = []
        for item in exp:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            desc = row.get("description")
            if isinstance(desc, list):
                row["description"] = "; ".join(str(x).strip() for x in desc if str(x).strip()) or None
            fixed.append(row)
        data["experience"] = fixed

    skills = data.get("skills")
    if isinstance(skills, list):
        data["skills"] = [str(s).strip() for s in skills if str(s).strip()]

    return ResumeExtraction(
        name=data.get("name"),
        email=data.get("email"),
        phone=data.get("phone"),
        phones=data.get("phones") or [],
        summary=data.get("summary"),
        experience=[ExperienceItem(**x) for x in (data.get("experience") or []) if isinstance(x, dict)],
        education=[EducationItem(**x) for x in (data.get("education") or []) if isinstance(x, dict)],
        skills=data.get("skills") or [],
    )


def _extract_resume_direct_openai(text: str, model_id: str) -> ResumeExtraction:
    """Direct LM Studio / OpenAI-compatible extraction using a schema-in-prompt approach."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai package is required. Run: pip install langextract[openai]"
        ) from e

    base_url = (
        os.environ.get("LM_STUDIO_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    ).strip()
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip() or "lm-studio"
    max_tokens = int((os.environ.get("LOCAL_MAX_OUTPUT_TOKENS") or "2000").strip())
    logger.info("using model_id=%s base_url=%s", model_id, base_url or "default")

    schema_json = json.dumps(ResumeExtraction.model_json_schema(), ensure_ascii=False)
    system_prompt = (
        "Extract structured resume information. "
        "Return valid JSON only, no markdown, no extra text."
    )
    user_prompt = (
        "Extract resume information from this text and follow the schema strictly.\n\n"
        f"Schema:\n{schema_json}\n\n"
        f"Resume text:\n{text}"
    )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/") if base_url else None,
    )
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("Model returned empty response")

    for candidate in _extract_json_candidates(content):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return _normalize_model_payload(parsed)
        except Exception:
            continue

    raise RuntimeError("Model response is not valid JSON")


def extract_resume(text: str) -> ResumeExtraction:
    """
    Extract structured resume data from plain text.

    Uses LM Studio OpenAI-compatible chat completion.
    Set LM_STUDIO_BASE_URL and OPENAI_API_KEY.
    """
    if not (text or "").strip():
        return ResumeExtraction()
    model_id = _get_model_id()
    return _extract_resume_direct_openai(text.strip(), model_id)
