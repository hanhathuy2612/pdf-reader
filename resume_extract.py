"""Minimal resume extraction via OpenAI-compatible chat completion."""

import json
import logging
import os
import re
from typing import Any

from schemas import EducationItem, ExperienceItem, ResumeExtraction

logger = logging.getLogger("resume_extract")

JsonDict = dict[str, Any]
ExperienceRow = dict[str, str | None]


def _get_model_id() -> str:
    model = (os.environ.get("MODEL_ID") or "").strip()
    if model:
        return model
    return "qwen2.5-14b-instruct-1m"


def _create_openai_client() -> tuple[Any, int | None]:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai package is required. Run: pip install langextract[openai]"
        ) from e

    base_url = (
        os.environ.get("LM_STUDIO_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or ""
    ).strip()
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip() or "lm-studio"
    raw_max_tokens = (os.environ.get("LOCAL_MAX_OUTPUT_TOKENS") or "").strip()
    max_tokens = int(raw_max_tokens) if raw_max_tokens else None
    client = OpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/") if base_url else None,
    )
    return client, max_tokens


def _request_chat_completion_with_json_schema_fallback(
    client: Any,
    request_args: JsonDict,
    schema_name: str = "resume_payload",
    schema: JsonDict | None = None,
) -> Any:
    """
    Send a chat.completions request.

    Some OpenAI-compatible backends return:
    "JSON schema is missing in json-mode request"
    unless response_format.json_schema is provided. In that case, retry once
    with a permissive JSON schema so existing prompt behavior remains intact.
    """
    try:
        return client.chat.completions.create(**request_args)
    except Exception as exc:
        message = str(exc).lower()
        if "json schema is missing in json-mode request" not in message:
            raise

        fallback_schema = schema or {"type": "object", "additionalProperties": True}
        retry_args = dict(request_args)
        retry_args["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": False,
                "schema": fallback_schema,
            },
        }
        return client.chat.completions.create(**retry_args)


_PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{7,}\d")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
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
_NOISE_COMPANY_VALUES = {
    "nanoc",
    "core skills",
    "public speaking",
    "technical writing",
    "community impact",
    "open-source projects",
    "programming/markup languages",
    "technologies",
    "tools",
    "processes",
    "natural languages",
    "other interests",
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


def _join_nonempty_text(items: Any, sep: str = "; ") -> str | None:
    if isinstance(items, str):
        value = items.strip()
        return value or None
    if not isinstance(items, list):
        return None
    values = [str(x).strip() for x in items if str(x).strip()]
    return sep.join(values) if values else None


def _experience_identity(row: dict[str, Any]) -> str:
    return (
        f"{(row.get('title') or '').strip().lower()}|"
        f"{(row.get('company') or '').strip().lower()}|"
        f"{(row.get('dates') or '').strip().lower()}"
    )


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


def _parse_json_dict_loose(text: str) -> JsonDict | None:
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


def _guess_name_from_text(text: str) -> str | None:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    for line in lines[:12]:
        if "@" in line or line.startswith(("http://", "https://")):
            continue
        if _is_phone_like(line):
            continue
        if re.search(r"\d", line):
            continue
        if len(line.split()) >= 2 and len(line) <= 60:
            return line.title()
    return None


def _build_raw_fallback_from_text(source_text: str) -> JsonDict:
    emails = _EMAIL_RE.findall(source_text or "")
    phones = [
        _normalize_phone(m.group(0)) for m in _PHONE_RE.finditer(source_text or "")
    ]
    phones = _dedup_keep_order([p for p in phones if _is_phone_like(p)])
    basics: JsonDict = {
        "name": _guess_name_from_text(source_text),
        "emails": _dedup_keep_order(emails),
        "phones": phones,
        "location": None,
        "summary": _extract_summary_from_text(source_text),
    }
    return {
        "basics": basics,
        "experience": [dict(x) for x in _extract_experience_headers(source_text)],
        "education": [],
        "skills": [],
        "projects": [],
        "publications": [],
        "awards": [],
        "certifications": [],
        "languages": [],
        "extra_sections": [],
    }


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
    if s.lower() in _KNOWN_SECTIONS:
        return False
    if s.lower() in _NOISE_COMPANY_VALUES:
        return False
    if any(ch in s for ch in [".", ",", ";", ":"]):
        return False
    words = s.split()
    if len(words) > 6:
        return False
    if _is_probable_role_title(s):
        return False
    return bool(re.search(r"[A-Za-z]", s))


def _extract_experience_headers(text: str) -> list[ExperienceRow]:
    rows: list[ExperienceRow] = []
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    for i, line in enumerate(lines):
        date_match = _DATE_RANGE_RE.search(line)
        if not date_match:
            continue
        # Fallback parser should only consume lines that are mostly a date range.
        if date_match.group(0).strip() != line.strip():
            continue

        company: str | None = None
        title: str | None = None

        # Prefer a title->company pair directly above the date line.
        for j in range(i - 2, max(-1, i - 12), -1):
            maybe_title = lines[j]
            if maybe_title.startswith(("[SECTION:", "- ")):
                continue
            if not _is_probable_role_title(maybe_title):
                continue
            maybe_company = lines[j + 1] if j + 1 < i else ""
            if _is_probable_company(maybe_company):
                title = maybe_title
                company = maybe_company
                break

        # Fallback: nearest valid company with title somewhere above.
        if not (title and company):
            for j in range(i - 1, max(-1, i - 8), -1):
                candidate = lines[j]
                if candidate.startswith(("[SECTION:", "- ")):
                    continue
                if _is_probable_company(candidate):
                    company = candidate
                    for k in range(j - 1, max(-1, j - 8), -1):
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

    unique: list[ExperienceRow] = []
    seen: set[str] = set()
    for row in rows:
        key = _experience_identity(row)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _repair_with_text_fallback(data: JsonDict, source_text: str) -> JsonDict:
    repaired = dict(data)
    if not repaired.get("summary"):
        fallback_summary = _extract_summary_from_text(source_text)
        if fallback_summary:
            repaired["summary"] = fallback_summary

    current_exp = repaired.get("experience")
    if not isinstance(current_exp, list):
        current_exp = []
    current_exp = [x for x in current_exp if isinstance(x, dict)]

    # Raw-first policy: if model already found enough experience rows, do not
    # inject heuristic fallback rows from noisy PDF text.
    if len(current_exp) >= 2:
        repaired["experience"] = current_exp
        return repaired

    extracted_exp = _extract_experience_headers(source_text)
    if not extracted_exp:
        repaired["experience"] = current_exp
        return repaired

    if not current_exp:
        repaired["experience"] = extracted_exp
        return repaired

    existing_keys = {_experience_identity(x) for x in current_exp}
    for row in extracted_exp:
        key = _experience_identity(row)
        if key not in existing_keys:
            current_exp.append(row)
    repaired["experience"] = current_exp

    return repaired


def _augment_raw_payload_with_text(raw_payload: JsonDict, source_text: str) -> JsonDict:
    """Improve raw payload recall using deterministic anchors from source text."""
    out = dict(raw_payload)
    basics = out.get("basics")
    if not isinstance(basics, dict):
        basics = {}
    if not basics.get("summary"):
        summary = _extract_summary_from_text(source_text)
        if summary:
            basics["summary"] = summary
    out["basics"] = basics

    raw_exp = out.get("experience")
    if not isinstance(raw_exp, list):
        raw_exp = []
    raw_exp = [x for x in raw_exp if isinstance(x, dict)]

    anchors = _extract_experience_headers(source_text)
    if not anchors:
        out["experience"] = raw_exp
        return out

    by_date: dict[str, ExperienceRow] = {}
    for row in anchors:
        dates = row.get("dates")
        if isinstance(dates, str) and dates.strip() and dates not in by_date:
            by_date[dates] = row

    fixed: list[JsonDict] = []
    for row in raw_exp:
        updated = dict(row)
        raw_dates = updated.get("dates")
        if isinstance(raw_dates, str) and raw_dates in by_date:
            anchor = by_date[raw_dates]
            raw_title = str(updated.get("title") or "").strip()
            raw_company = str(updated.get("company") or "").strip()
            if not _is_probable_role_title(raw_title):
                updated["title"] = anchor.get("title")
            if not _is_probable_company(raw_company):
                updated["company"] = anchor.get("company")
        fixed.append(updated)

    existing = {_experience_identity(x) for x in fixed}
    for row in anchors:
        key = _experience_identity(row)
        if key not in existing:
            fixed.append(dict(row))

    out["experience"] = fixed
    return out


def _pick_first_nonempty_str(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _to_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _coerce_raw_to_schema_dict(payload: JsonDict) -> JsonDict:
    """
    Convert a free-form/raw resume JSON into the project schema shape.
    Keeps behavior stable for both strict-schema and arbitrary-schema model outputs.
    """
    data = dict(payload)
    basics = data.get("basics") if isinstance(data.get("basics"), dict) else {}
    contact = data.get("contact") if isinstance(data.get("contact"), dict) else {}
    basics_emails = _to_string_list(basics.get("emails"))
    contact_emails = _to_string_list(contact.get("emails"))

    phones = _to_string_list(data.get("phones"))
    phones.extend(_to_string_list(basics.get("phones")))
    phones.extend(_to_string_list(contact.get("phones")))
    phones.extend(_to_string_list(contact.get("phone_numbers")))
    primary_phone = _pick_first_nonempty_str(
        data.get("phone"), basics.get("phone"), contact.get("phone")
    )
    if primary_phone:
        phones.insert(0, primary_phone)

    exp_source = (
        data.get("experience")
        or data.get("work_experience")
        or data.get("work")
        or data.get("employment")
        or []
    )
    experience: list[dict[str, Any]] = []
    if isinstance(exp_source, list):
        for row in exp_source:
            if not isinstance(row, dict):
                continue
            desc = _join_nonempty_text(row.get("description"))
            if not desc:
                highlights = (
                    row.get("highlights")
                    or row.get("achievements")
                    or row.get("responsibilities")
                )
                desc = _join_nonempty_text(highlights)
            item = {
                "title": _pick_first_nonempty_str(
                    row.get("title"),
                    row.get("position"),
                    row.get("role"),
                    row.get("job_title"),
                ),
                "company": _pick_first_nonempty_str(
                    row.get("company"),
                    row.get("employer"),
                    row.get("organization"),
                    row.get("org"),
                ),
                "dates": _pick_first_nonempty_str(
                    row.get("dates"),
                    row.get("date_range"),
                    row.get("period"),
                    row.get("duration"),
                ),
                "description": desc if isinstance(desc, str) else None,
            }
            if item["title"] and item["company"]:
                experience.append(item)

    edu_source = data.get("education") or data.get("academic") or []
    education: list[dict[str, Any]] = []
    if isinstance(edu_source, list):
        for row in edu_source:
            if not isinstance(row, dict):
                continue
            item = {
                "degree": _pick_first_nonempty_str(
                    row.get("degree"),
                    row.get("qualification"),
                    row.get("program"),
                    row.get("title"),
                ),
                "institution": _pick_first_nonempty_str(
                    row.get("institution"),
                    row.get("school"),
                    row.get("university"),
                    row.get("organization"),
                ),
                "dates": _pick_first_nonempty_str(
                    row.get("dates"),
                    row.get("date_range"),
                    row.get("period"),
                    row.get("duration"),
                ),
            }
            if item["degree"] and item["institution"]:
                education.append(item)

    skills_source = (
        data.get("skills")
        or data.get("core_skills")
        or data.get("technical_skills")
        or []
    )
    skills: list[str] = []
    if isinstance(skills_source, list):
        for row in skills_source:
            if isinstance(row, str) and row.strip():
                skills.append(row.strip())
            elif isinstance(row, dict):
                skills.extend(_to_string_list(row.get("items")))
    elif isinstance(skills_source, dict):
        for value in skills_source.values():
            skills.extend(_to_string_list(value))

    return {
        "name": _pick_first_nonempty_str(data.get("name"), basics.get("name")),
        "email": _pick_first_nonempty_str(
            data.get("email"),
            basics.get("email"),
            contact.get("email"),
            basics_emails[0] if basics_emails else None,
            contact_emails[0] if contact_emails else None,
        ),
        "phone": primary_phone,
        "phones": phones,
        "summary": _pick_first_nonempty_str(
            data.get("summary"),
            data.get("profile"),
            data.get("objective"),
            data.get("about"),
            basics.get("summary"),
        ),
        "experience": experience or data.get("experience") or [],
        "education": education or data.get("education") or [],
        "skills": skills or _to_string_list(data.get("skills")),
    }


def _normalize_model_payload(
    payload: JsonDict, source_text: str, apply_text_fallback: bool = True
) -> ResumeExtraction:
    """Lightweight normalization only — keep model output as-is as much as possible."""
    data: dict[str, Any] = dict(payload)

    # Some models wrap with {"result": {...}}
    if "result" in data and isinstance(data["result"], dict):
        data = dict(data["result"])

    data = _coerce_raw_to_schema_dict(data)
    if apply_text_fallback:
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
                row["description"] = _join_nonempty_text(desc)
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
        experience=[
            ExperienceItem(**x)
            for x in (data.get("experience") or [])
            if isinstance(x, dict)
        ],
        education=[
            EducationItem(**x)
            for x in (data.get("education") or [])
            if isinstance(x, dict)
        ],
        skills=data.get("skills") or [],
    )


def _extract_raw_resume_json_openai(text: str, model_id: str) -> JsonDict:
    """Step 1: ask model to return a rich, free-form resume JSON (lossless-first)."""
    client, max_tokens = _create_openai_client()
    logger.info("using model_id=%s for raw extraction", model_id)

    system_prompt = (
        "You extract resume information from noisy PDF text. "
        "Return one valid JSON object only (no markdown, no prose)."
    )
    user_prompt = (
        "Extract as much information as possible from this resume text.\n"
        "Return one JSON object with your own best structure.\n"
        "Do not lose data and do not add markdown/prose.\n"
        "Use null/empty arrays only when information is missing.\n\n"
        f"Resume text:\n{text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    request_args: JsonDict = {
        "model": model_id,
        "messages": messages,
        "temperature": 0,
    }
    if max_tokens is not None:
        request_args["max_tokens"] = max_tokens
    resp = _request_chat_completion_with_json_schema_fallback(
        client,
        request_args,
        schema_name="resume_raw_payload",
    )

    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("Model returned empty response")

    parsed = _parse_json_dict_loose(content)
    if parsed is not None:
        return parsed

    # Retry once with explicit JSON-fix instruction for smaller models.
    repair_prompt = (
        "Rewrite the previous assistant output as strict valid JSON only. "
        "Do not add or remove keys. Return one JSON object."
    )
    repair_args: JsonDict = {
        "model": model_id,
        "messages": [
            *messages,
            {"role": "assistant", "content": content},
            {"role": "user", "content": repair_prompt},
        ],
        "temperature": 0,
    }
    if max_tokens is not None:
        repair_args["max_tokens"] = max_tokens
    repair_resp = _request_chat_completion_with_json_schema_fallback(
        client,
        repair_args,
        schema_name="resume_raw_payload_repair",
    )
    repaired_content = (repair_resp.choices[0].message.content or "").strip()
    repaired = _parse_json_dict_loose(repaired_content)
    if repaired is not None:
        return repaired

    raise RuntimeError("Model response is not valid JSON")


def _normalize_raw_to_schema_openai(
    raw_payload: JsonDict, source_text: str, model_id: str
) -> JsonDict:
    """Step 2: normalize raw JSON to strict project schema via prompt."""
    client, max_tokens = _create_openai_client()
    schema_guide = {
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
        "education": [
            {"degree": "string", "institution": "string", "dates": "string|null"}
        ],
        "skills": ["string"],
    }
    system_prompt = (
        "You normalize resume data into a strict target JSON schema. "
        "Return one valid JSON object only."
    )
    user_prompt = (
        "Map RAW_JSON into TARGET_SCHEMA.\n"
        "Rules:\n"
        "1) Keep as much mappable information as possible.\n"
        "2) Do not invent facts. Use SOURCE_TEXT only for disambiguation.\n"
        "3) Preserve all valid experience and education entries.\n"
        "4) If an entry misses required fields and cannot be inferred, drop that entry only.\n"
        "5) Return strict JSON only.\n\n"
        f"TARGET_SCHEMA:\n{json.dumps(schema_guide, ensure_ascii=False)}\n\n"
        f"RAW_JSON:\n{json.dumps(raw_payload, ensure_ascii=False)}\n\n"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    request_args: JsonDict = {
        "model": model_id,
        "messages": messages,
        "temperature": 0,
    }
    if max_tokens is not None:
        request_args["max_tokens"] = max_tokens
    resp = _request_chat_completion_with_json_schema_fallback(
        client,
        request_args,
        schema_name="resume_schema_payload",
    )
    content = (resp.choices[0].message.content or "").strip()
    parsed = _parse_json_dict_loose(content)
    if parsed is not None:
        return parsed

    repair_prompt = (
        "Rewrite the previous assistant output as strict valid JSON for TARGET_SCHEMA. "
        "Return one JSON object only."
    )
    repair_args: JsonDict = {
        "model": model_id,
        "messages": [
            *messages,
            {"role": "assistant", "content": content},
            {"role": "user", "content": repair_prompt},
        ],
        "temperature": 0,
    }
    if max_tokens is not None:
        repair_args["max_tokens"] = max_tokens
    repair_resp = _request_chat_completion_with_json_schema_fallback(
        client,
        repair_args,
        schema_name="resume_schema_payload_repair",
    )
    repaired = _parse_json_dict_loose(
        (repair_resp.choices[0].message.content or "").strip()
    )
    if repaired is not None:
        return repaired
    raise RuntimeError("Model response is not valid JSON")


def extract_resume_with_raw(text: str) -> tuple[ResumeExtraction, dict[str, Any]]:
    """
    Two-step extraction:
    1) Model produces a rich/free-form raw JSON (max information retention).
    2) Prompt-based normalization maps raw JSON to project schema.
    """
    if not (text or "").strip():
        return ResumeExtraction(), {}

    model_id = _get_model_id()
    preprocessed_text = _preprocess_resume_text(text.strip())
    raw_payload = _extract_raw_resume_json_openai(preprocessed_text, model_id)
    try:
        normalized_payload = _normalize_raw_to_schema_openai(
            raw_payload, preprocessed_text, model_id
        )
        normalized = _normalize_model_payload(
            normalized_payload, preprocessed_text, apply_text_fallback=True
        )
    except Exception:
        # Safe fallback if step-2 model normalization fails.
        normalized = _normalize_model_payload(raw_payload, preprocessed_text)
    return normalized, raw_payload


def extract_resume(text: str) -> ResumeExtraction:
    """
    Extract structured resume data from plain text.

    Uses LM Studio OpenAI-compatible chat completion.
    Set LM_STUDIO_BASE_URL and OPENAI_API_KEY.
    """
    if not (text or "").strip():
        return ResumeExtraction()
    result, _ = extract_resume_with_raw(text)
    return result
