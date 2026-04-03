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
_BULLET_PREFIX_RE = re.compile(r"^[•●▪◦‣]\s*")
_MONTH_TOKEN = (
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|"
    r"January|February|March|April|May|June|July|August|September|October|November|December)"
)
_DATE_RANGE_RE = re.compile(
    rf"(?:{_MONTH_TOKEN}\s+\d{{4}}\s*[–-]\s*(?:{_MONTH_TOKEN}\s+\d{{4}}|Present))|(?:…\s*[–-]\s*\d{{4}})"
)
_KNOWN_SECTIONS: dict[str, str] = {
    "work experience": "WORK_EXPERIENCE",
    "experience": "WORK_EXPERIENCE",
    "education": "EDUCATION",
    "core skills": "SKILLS",
    "skills": "SKILLS",
    "technical writing": "TECHNICAL_WRITING",
    "community impact": "COMMUNITY_IMPACT",
    "public speaking": "PUBLIC_SPEAKING",
}


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


def _extract_balanced_json_object(text: str) -> str | None:
    s = text or ""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _parse_json_dict_loose(text: str) -> dict[str, Any] | None:
    for candidate in _extract_json_candidates(text):
        attempt = candidate.strip()
        if not attempt:
            continue
        raw_attempts = [
            attempt,
            re.sub(r",\s*([}\]])", r"\1", attempt),  # remove trailing commas
        ]
        balanced = _extract_balanced_json_object(attempt)
        if balanced:
            raw_attempts.extend(
                [
                    balanced,
                    re.sub(r",\s*([}\]])", r"\1", balanced),
                ]
            )

        for raw in raw_attempts:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
    return None


def _looks_like_standalone_label(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if re.search(r"[.:;!?]", s):
        return False
    words = s.split()
    if len(words) > 6:
        return False
    titled = 0
    for w in words:
        token = re.sub(r"[^A-Za-z0-9/&()\-]", "", w)
        if not token:
            continue
        if token.isupper() or re.match(r"^[A-Z][a-z].*", token):
            titled += 1
    return titled >= max(1, len(words) - 1)


def _normalize_line(line: str) -> str:
    s = (line or "").strip()
    if not s:
        return ""
    # Normalize common PDF artifacts and Unicode bullets.
    s = s.replace("\u00ad", "")  # soft hyphen
    s = s.replace("\u2010", "-").replace("\u2011", "-").replace("\u2212", "-")
    s = _BULLET_PREFIX_RE.sub("- ", s)
    if s in {"•", "●", "▪", "◦", "‣", "-"}:
        return "-"
    return re.sub(r"\s+", " ", s).strip()


def _is_section_heading(line: str) -> str | None:
    key = (line or "").strip().lower()
    return _KNOWN_SECTIONS.get(key)


def _preprocess_resume_text(text: str) -> str:
    lines = [_normalize_line(line) for line in (text or "").splitlines()]
    out: list[str] = []

    for line in lines:
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue

        section = _is_section_heading(line)
        if section:
            if out and out[-1] != "":
                out.append("")
            out.append(f"[SECTION: {section}]")
            out.append(line)
            continue

        if not out or out[-1] == "":
            out.append(line)
            continue

        prev = out[-1]
        is_prev_bullet = prev.startswith("- ")
        is_curr_bullet = line.startswith("- ")
        should_join = (
            not is_prev_bullet
            and not is_curr_bullet
            and not prev.endswith((".", "!", "?", ":", ";"))
            and not _looks_like_standalone_label(prev)
            and not _looks_like_standalone_label(line)
        )

        if should_join:
            # Preserve hyphenated words split across line breaks.
            if prev.endswith("-"):
                out[-1] = prev[:-1] + line
            else:
                out[-1] = f"{prev} {line}"
        else:
            out.append(line)

    processed = "\n".join(out).strip()
    return processed or (text or "").strip()


def _extract_summary_from_text(text: str) -> str | None:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    for line in lines:
        low = line.lower()
        if low in {
            "work experience",
            "education",
            "core skills",
            "skills",
            "community impact",
            "public speaking",
            "technical writing",
        }:
            break
        if (
            len(line) >= 50
            and "@" not in line
            and not _is_phone_like(line)
            and not line.startswith(("http://", "https://"))
        ):
            return line
    return None


def _is_probable_role_title(line: str) -> bool:
    s = (line or "").strip()
    if not s or len(s) > 80:
        return False
    low = s.lower()
    keywords = (
        "engineer",
        "developer",
        "manager",
        "lecturer",
        "coordinator",
        "analyst",
        "consultant",
        "architect",
        "intern",
    )
    return any(k in low for k in keywords)


def _is_probable_company(line: str) -> bool:
    s = (line or "").strip()
    if not s or len(s) > 80:
        return False
    if any(ch in s for ch in [".", ",", ";", ":"]):
        return False
    words = s.split()
    if len(words) > 6:
        return False
    if _is_probable_role_title(s):
        return False
    return bool(re.search(r"[A-Za-z]", s))


def _extract_experience_headers(text: str) -> list[dict[str, str | None]]:
    rows: list[dict[str, str | None]] = []
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    for i, line in enumerate(lines):
        if not _DATE_RANGE_RE.search(line):
            continue

        company: str | None = None
        title: str | None = None

        for j in range(i - 1, max(-1, i - 7), -1):
            candidate = lines[j]
            if candidate.startswith(("[SECTION:", "- ")):
                continue
            if _is_probable_company(candidate):
                company = candidate
                for k in range(j - 1, max(-1, j - 6), -1):
                    maybe_title = lines[k]
                    if maybe_title.startswith(("[SECTION:", "- ")):
                        continue
                    if _is_probable_role_title(maybe_title):
                        title = maybe_title
                        break
                break

        if title and company:
            rows.append(
                {
                    "title": title,
                    "company": company,
                    "dates": line,
                    "description": None,
                }
            )

    unique: list[dict[str, str | None]] = []
    seen: set[str] = set()
    for row in rows:
        key = f"{row['title']}|{row['company']}|{row['dates']}".lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _repair_with_text_fallback(data: dict[str, Any], source_text: str) -> dict[str, Any]:
    repaired = dict(data)
    if not repaired.get("summary"):
        fallback_summary = _extract_summary_from_text(source_text)
        if fallback_summary:
            repaired["summary"] = fallback_summary

    extracted_exp = _extract_experience_headers(source_text)
    current_exp = repaired.get("experience")
    if not isinstance(current_exp, list):
        current_exp = []
    current_exp = [x for x in current_exp if isinstance(x, dict)]

    if len(current_exp) < len(extracted_exp):
        existing_keys = {
            f"{(x.get('title') or '').strip().lower()}|{(x.get('company') or '').strip().lower()}|{(x.get('dates') or '').strip().lower()}"
            for x in current_exp
        }
        for row in extracted_exp:
            key = f"{(row.get('title') or '').strip().lower()}|{(row.get('company') or '').strip().lower()}|{(row.get('dates') or '').strip().lower()}"
            if key not in existing_keys:
                current_exp.append(row)
        repaired["experience"] = current_exp

    return repaired


def _normalize_model_payload(payload: dict[str, Any], source_text: str) -> ResumeExtraction:
    """Lightweight normalization only — keep model output as-is as much as possible."""
    data: dict[str, Any] = dict(payload)

    # Some models wrap with {"result": {...}}
    if "result" in data and isinstance(data["result"], dict):
        data = dict(data["result"])

    data = _repair_with_text_fallback(data, source_text)

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

    compact_schema = {
        "name": "string|null",
        "email": "string|null",
        "phone": "string|null",
        "phones": ["string"],
        "summary": "string|null",
        "experience": [
            {
                "title": "string",
                "company": "string",
                "dates": "string|null",
                "description": "string|null",
            }
        ],
        "education": [{"degree": "string", "institution": "string", "dates": "string|null"}],
        "skills": ["string"],
    }
    system_prompt = (
        "You extract structured resume information from noisy PDF text. "
        "Return valid JSON only (no markdown, no prose). "
        "Follow the schema exactly and preserve the original structure."
    )
    extraction_rules = (
        "Extraction rules:\n"
        "1) Keep the output strictly schema-compliant.\n"
        "2) For each experience item, always capture title/company/dates when present.\n"
        "3) Assign achievement/responsibility bullets only when role linkage is clear.\n"
        "4) If role linkage is ambiguous, keep description as null instead of guessing.\n"
        "5) Join multiple bullet lines into one concise description string when assigned.\n"
        "6) Prefer correctness of mapping over completeness of descriptions.\n"
        "7) Do not invent facts not present in the resume text.\n"
    )
    user_prompt = (
        "Extract resume information from this text and follow the schema strictly.\n\n"
        f"{extraction_rules}\n"
        f"Schema:\n{json.dumps(compact_schema, ensure_ascii=False)}\n\n"
        f"Resume text:\n{text}"
    )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/") if base_url else None,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
    )
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("Model returned empty response")

    parsed = _parse_json_dict_loose(content)
    if parsed is not None:
        return _normalize_model_payload(parsed, text)

    # Retry once with explicit JSON-fix instruction for smaller models.
    repair_prompt = (
        "Rewrite the previous assistant output as strict valid JSON only. "
        "Do not add or remove keys. Return one JSON object."
    )
    repair_resp = client.chat.completions.create(
        model=model_id,
        messages=[
            *messages,
            {"role": "assistant", "content": content},
            {"role": "user", "content": repair_prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    repaired_content = (repair_resp.choices[0].message.content or "").strip()
    repaired = _parse_json_dict_loose(repaired_content)
    if repaired is not None:
        return _normalize_model_payload(repaired, text)

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
    preprocessed_text = _preprocess_resume_text(text.strip())
    return _extract_resume_direct_openai(preprocessed_text, model_id)
