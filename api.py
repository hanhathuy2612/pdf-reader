"""FastAPI app: PDF upload and resume extraction as JSON."""

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ImportError:
    pass

# Enable debug logging when DEBUG=1 or DEBUG=true in env
if os.environ.get("DEBUG", "").strip().lower() in ("1", "true", "yes"):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("resume_extract").setLevel(logging.DEBUG)

from fastapi import FastAPI, File, HTTPException, UploadFile

from pdf_reader import extract_text_from_pdf
from resume_extract import extract_resume_with_raw
from schemas import ResumeExtraction

logger = logging.getLogger("api")

app = FastAPI(
    title="ATS PDF Reader",
    description="Upload a resume PDF and receive structured JSON (name, experience, education, skills).",
    version="1.0.0",
)

# Limit upload size (e.g. 20 MB)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ARTIFACTS_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = ARTIFACTS_DIR / "result"


def _slugify_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())
    safe = safe.strip("._-")
    return safe or "resume"


def _build_result_paths(original_filename: str) -> tuple[Path, Path, Path]:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    stem = _slugify_filename(Path(original_filename).stem)
    input_path = run_dir / f"{stem}_input-pdf.txt"
    raw_path = run_dir / f"{stem}_raw-result.json"
    result_path = run_dir / f"{stem}_result.json"
    return input_path, raw_path, result_path


def _write_debug_artifacts(
    paths: tuple[Path, Path, Path],
    extracted_text: str,
    raw_result: dict[str, Any] | None = None,
    result: ResumeExtraction | None = None,
) -> None:
    """Persist extracted text and JSON result under a per-run timestamp folder."""
    input_path, raw_path, result_path = paths
    input_path.write_text(extracted_text or "", encoding="utf-8")
    if raw_result is not None:
        raw_path.write_text(
            json.dumps(raw_result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if result is not None:
        result_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: API is up."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    """
    Readiness for LM Studio OpenAI-compatible endpoint.
    """
    import urllib.request

    base_url = (
        os.environ.get("LM_STUDIO_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "http://127.0.0.1:1234/v1"
    ).strip()
    health_url = f"{base_url.rstrip('/')}/models"

    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as _:
            pass
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "lmstudio_unreachable",
                "error": str(e),
                "url": health_url,
            },
        )

    return {
        "status": "ok",
        "provider": "openai-compatible",
        "base_url": base_url,
        "api_key_set": bool((os.environ.get("OPENAI_API_KEY") or "").strip()),
    }


@app.post("/extract", response_model=ResumeExtraction)
async def extract(
    file: UploadFile = File(..., description="Resume PDF file")
) -> ResumeExtraction:
    """
    Upload a resume PDF and get structured JSON (name, contact, experience, education, skills).
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="File must be a PDF (e.g. resume.pdf)"
        )

    t0 = time.perf_counter()
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB)",
        )
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        t_pdf_start = time.perf_counter()
        text = extract_text_from_pdf(raw)
        t_pdf_s = time.perf_counter() - t_pdf_start
        artifact_paths = _build_result_paths(file.filename)
        _write_debug_artifacts(artifact_paths, text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid PDF: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read PDF: {e}")

    try:
        t_llm_start = time.perf_counter()
        result, raw_result = extract_resume_with_raw(text)
        t_llm_s = time.perf_counter() - t_llm_start
        _write_debug_artifacts(artifact_paths, text, raw_result, result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    t_total_s = time.perf_counter() - t0
    logger.info(
        (
            "extract completed | file=%s | bytes=%d | chars=%d | "
            "timings={pdf: %.3fs, llm: %.3fs, total: %.3fs}"
        ),
        file.filename,
        len(raw),
        len(text or ""),
        t_pdf_s,
        t_llm_s,
        t_total_s,
    )

    return result


def run_app() -> None:
    """Run uvicorn (host/port from env for deployment)."""
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=os.environ.get("RELOAD", "").lower() in ("1", "true"),
    )
