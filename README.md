# pdf-reader (CV from PDF)

Small API: accepts a **CV as PDF or raster image** (PNG, JPEG, WebP, GIF, BMP, multi-page TIFF, …). PDFs are rendered to PNG per page; images are normalized to PNG. Those images are sent to LM Studio (OpenAI-compatible); response is structured JSON (name, email, phone, skills, education, experience).

> **Note — VL model required.** This app sends **images** (PNG pages), not raw PDF text. You **must** load a **vision-language (VL)** model in LM Studio (e.g. Qwen2.5-VL, LLaVA). A text-only instruct model will not read the images and extraction will fail or be useless.

## Requirements

- [LM Studio](https://lmstudio.ai/) with the local server running (default `http://localhost:1234/v1`).
- A **VL** model loaded and selected (e.g. Qwen2.5-VL Instruct). Text-only models are **not** supported for this endpoint.

## Setup and run

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` (example):

```env
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio
MODEL_ID=<VL model id — must match the vision model loaded in LM Studio>
```

Start the server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API

`POST /parse-cv` — `multipart/form-data` with field `file`: a **PDF** or a **raster image** (e.g. a phone photo of a resume).

Examples:

```bash
curl -s -X POST "http://localhost:8000/parse-cv" -F "file=@./resume.pdf"
curl -s -X POST "http://localhost:8000/parse-cv" -F "file=@./resume.png"
```

## Environment variables (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `qwen2.5-7b-instruct` | **Use a VL model id** (must match LM Studio). Default is text-only — change it. |
| `OPENAI_BASE_URL` | `http://localhost:1234/v1` | API base URL |
| `OPENAI_API_KEY` | `lm-studio` | API key (LM Studio is often lenient) |
| `PDF_MAX_PAGES` | `15` | Max PDF pages **or** image frames (e.g. TIFF) sent to the model |
| `PDF_RENDER_ZOOM` | `2.0` | PDF render scale only (lower if you hit context limits) |
| `LOG_LEVEL` | `INFO` | Logging level |

If the document is long or you get context errors, lower `PDF_MAX_PAGES` or `PDF_RENDER_ZOOM` (PDF only).
