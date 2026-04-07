from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from openai import OpenAI
from PIL import Image, ImageSequence
from pydantic import BaseModel
import base64
import json
import logging
import os
import time
from dotenv import load_dotenv

import fitz  # PyMuPDF

app = FastAPI()
logger = logging.getLogger("cv_parser")

load_dotenv()

LOG_LEVEL = (os.environ.get("LOG_LEVEL") or "INFO").upper().strip()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

MODEL_ID = (os.environ.get("MODEL_ID") or "qwen2.5-7b-instruct").strip()
OPENAI_BASE_URL = (
    os.environ.get("OPENAI_BASE_URL") or "http://localhost:1234/v1"
).strip()
OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "lm-studio").strip()

PDF_MAX_PAGES = int((os.environ.get("PDF_MAX_PAGES") or "15").strip() or "15")
PDF_RENDER_ZOOM = float((os.environ.get("PDF_RENDER_ZOOM") or "2.0").strip() or "2.0")

# LM Studio OpenAI-compatible client
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)


CV_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "cv_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "email": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "phone": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "skills": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ],
                    "items": {"type": "string"},
                },
                "education": {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "school": {
                                        "anyOf": [
                                            {"type": "string"},
                                            {"type": "null"},
                                        ]
                                    },
                                    "degree": {
                                        "anyOf": [
                                            {"type": "string"},
                                            {"type": "null"},
                                        ]
                                    },
                                    "year": {
                                        "anyOf": [
                                            {"type": "string"},
                                            {"type": "null"},
                                        ]
                                    },
                                },
                                "required": ["school", "degree", "year"],
                                "additionalProperties": True,
                            },
                        },
                        {"type": "null"},
                    ],
                    "items": {
                        "type": "object",
                        "properties": {
                            "school": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "degree": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "year": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        },
                        "required": ["school", "degree", "year"],
                        "additionalProperties": True,
                    },
                },
                "experience": {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "company": {
                                        "anyOf": [
                                            {"type": "string"},
                                            {"type": "null"},
                                        ]
                                    },
                                    "role": {
                                        "anyOf": [
                                            {"type": "string"},
                                            {"type": "null"},
                                        ]
                                    },
                                    "duration": {
                                        "anyOf": [
                                            {"type": "string"},
                                            {"type": "null"},
                                        ]
                                    },
                                    "description": {
                                        "anyOf": [
                                            {"type": "string"},
                                            {"type": "null"},
                                        ]
                                    },
                                },
                                "required": [
                                    "company",
                                    "role",
                                    "duration",
                                    "description",
                                ],
                                "additionalProperties": True,
                            },
                        },
                        {"type": "null"},
                    ],
                    "items": {
                        "type": "object",
                        "properties": {
                            "company": {
                                "anyOf": [{"type": "string"}, {"type": "null"}]
                            },
                            "role": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "duration": {
                                "anyOf": [{"type": "string"}, {"type": "null"}]
                            },
                            "description": {
                                "anyOf": [{"type": "string"}, {"type": "null"}]
                            },
                        },
                        "required": ["company", "role", "duration", "description"],
                        "additionalProperties": True,
                    },
                },
            },
            "required": ["name", "email", "phone", "skills", "education", "experience"],
            "additionalProperties": False,
        },
    },
}


class CV(BaseModel):
    name: str | None
    email: str | None
    phone: str | None
    skills: list[str] | None
    education: list[dict] | None
    experience: list[dict] | None


def pdf_to_png_data_urls(pdf_bytes: bytes) -> list[str]:
    """Render each PDF page to PNG and return data:image/png;base64,... URLs."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        n = min(doc.page_count, max(1, PDF_MAX_PAGES))
        mat = fitz.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM)
        urls: list[str] = []
        for i in range(n):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            png_bytes = pix.tobytes("png")
            b64 = base64.b64encode(png_bytes).decode("ascii")
            urls.append(f"data:image/png;base64,{b64}")
        return urls
    finally:
        doc.close()


def _frame_to_png_data_url(frame: Image.Image) -> str:
    rgb = frame.convert("RGB")
    buf = BytesIO()
    rgb.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def raster_image_to_png_data_urls(image_bytes: bytes) -> list[str]:
    """Decode raster image(s) (PNG, JPEG, WebP, GIF, BMP, multi-page TIFF, …) to PNG data URLs."""
    try:
        im = Image.open(BytesIO(image_bytes))
        im.load()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode image: {e}",
        ) from e

    urls: list[str] = []
    for i, frame in enumerate(ImageSequence.Iterator(im)):
        if i >= PDF_MAX_PAGES:
            break
        urls.append(_frame_to_png_data_url(frame))

    if not urls:
        raise HTTPException(status_code=400, detail="Image has no frames")

    return urls


def bytes_to_resume_image_data_urls(raw: bytes) -> tuple[list[str], str]:
    """
    Build model-ready image data URLs from PDF bytes or raster image bytes.
    Returns (urls, source_label) where source_label is 'pdf' or 'image'.
    """
    if len(raw) >= 4 and raw[:4] == b"%PDF":
        try:
            return pdf_to_png_data_urls(raw), "pdf"
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid or unreadable PDF: {e}",
            ) from e
    return raster_image_to_png_data_urls(raw), "image"


def call_llm_with_resume_images(
    image_data_urls: list[str],
    filename: str,
    source: str,
) -> str:
    source_note = (
        "The images are rendered pages of a PDF resume"
        if source == "pdf"
        else "The images are from an uploaded resume image file (one or more frames/pages)"
    )
    instruction = f"""You are an expert in extracting structured data from resume/CV documents.

{source_note} (file: {filename}), in order (page 1 first).

Extract:
- name
- email
- phone
- skills (array)
- education (array of {{school, degree, year}})
- experience (array of {{company, role, duration, description}})

Rules:
- Read text from the images (including scanned PDFs).
- Return ONLY valid JSON matching the enforced schema.
- No explanation or markdown.
- Missing field → null
"""

    content: list[dict] = [{"type": "text", "text": instruction}]
    for idx, url in enumerate(image_data_urls):
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": url},
            }
        )
        # tiny label helps some models separate pages
        content.append(
            {
                "type": "text",
                "text": f"(image above: page {idx + 1} of {len(image_data_urls)})",
            }
        )

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": "You are a JSON extraction engine. Use the resume page images.",
            },
            {"role": "user", "content": content},
        ],
        response_format=CV_RESPONSE_FORMAT,
        temperature=0,
    )

    return response.choices[0].message.content or ""


def safe_json_parse(content: str):
    try:
        return json.loads(content)
    except Exception:
        fix_prompt = f"Fix this JSON:\n{content}"
        retry = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": fix_prompt}],
            response_format=CV_RESPONSE_FORMAT,
            temperature=0,
        )
        return json.loads(retry.choices[0].message.content or "{}")


@app.post("/parse-cv")
async def parse_cv(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    t_render = time.perf_counter()
    image_urls, source = bytes_to_resume_image_data_urls(raw)
    render_s = time.perf_counter() - t_render

    t_llm = time.perf_counter()
    llm_output = call_llm_with_resume_images(
        image_urls,
        file.filename or f"resume.{source}",
        source,
    )
    llm_s = time.perf_counter() - t_llm

    data = safe_json_parse(llm_output)
    cv = CV(**data)

    total_seconds = time.perf_counter() - t0
    logger.info(
        "parse-cv completed | file=%s | source=%s | model=%s | images=%d | "
        "timings={render: %.3fs, llm: %.3fs, total: %.3fs, total_minutes: %.2f}",
        file.filename,
        source,
        MODEL_ID,
        len(image_urls),
        render_s,
        llm_s,
        total_seconds,
        total_seconds / 60.0,
    )

    return cv
