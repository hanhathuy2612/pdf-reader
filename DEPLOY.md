# Deployment Guide — ATS PDF Reader

This document describes how to deploy the ATS PDF Reader API (PDF upload → resume JSON) using Docker or a local Python environment.

## Prerequisites

- **Option A — Docker:** Docker and (optional) Docker Compose installed.
- **Option B — Without Docker (Ollama):** Python 3.10+ and [Ollama](https://ollama.com/) installed and running.
- **Option C — API key (no Ollama):** Python 3.10+ and a **Google Gemini** or **OpenAI** API key. Set `LANGEXTRACT_API_KEY` or `OPENAI_API_KEY`; no local model required.

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| **Provider selection** | If `OPENAI_API_KEY` is set → OpenAI. Else if `LANGEXTRACT_API_KEY` is set → Gemini. Else → Ollama. | — |
| `LANGEXTRACT_API_KEY` | Google Gemini API key ([Get one](https://aistudio.google.com/)). When set, extraction uses Gemini instead of Ollama. | — |
| `OPENAI_API_KEY` | OpenAI API key. When set, extraction uses OpenAI (e.g. gpt-4o) instead of Ollama. Requires `langextract[openai]`. | — |
| `OLLAMA_BASE_URL` | Base URL of the Ollama server (used only when no API key is set). | `http://localhost:11434` |
| `MODEL_ID` | Model for extraction. Ollama: e.g. `llama3.2` (must be pulled). Gemini: e.g. `gemini-2.5-flash`. OpenAI: e.g. `gpt-4o`. | `llama3.2` / `gemini-2.5-flash` / `gpt-4o` per provider |
| `MAX_CHAR_BUFFER` | Max characters per text chunk. Larger (e.g. 4000–5000) reduces "Content must contain an extractions key" when a chunk has no entities. | `4000` |
| `HOST` | Bind address for the API server. | `0.0.0.0` |
| `PORT` | Port for the API server. | `8000` |
| `DEBUG` | Set to `1` or `true` to enable debug logging (LangExtract + app). Noisy; use only for troubleshooting. | — |

Set these in your shell, in a `.env` file (when using Docker Compose), or in your container/orchestrator config. Copy `.env.example` to `.env` and adjust as needed.

**When using Ollama:** The extraction model must be **pulled** before the first `/extract` call, or you will get: `Extraction failed: Ollama API error: Can't find Ollama llama3.2`. See [Troubleshooting](#troubleshooting) and the deploy steps below.

---

## Deploy with API key (Gemini or OpenAI)

You can use a cloud API instead of local Ollama. No Docker or Ollama setup required.

1. Install dependencies (includes OpenAI support):
   ```bash
   pip install -r requirements.txt
   ```

2. Set **one** of the following (not both for the same run):
   - **Gemini:** `export LANGEXTRACT_API_KEY=your-gemini-api-key`  
     Get a key at [Google AI Studio](https://aistudio.google.com/).
   - **OpenAI:** `export OPENAI_API_KEY=sk-your-openai-key`

3. Optional: set `MODEL_ID` (e.g. `gemini-2.5-flash`, `gpt-4o`, or `gpt-4o-mini`).

4. Run the API:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

5. **Docker with API key:** build and run with the key passed as an env var:
   ```bash
   docker build -t ats-pdf-reader .
   docker run -p 8000:8000 -e LANGEXTRACT_API_KEY=your-key ats-pdf-reader
   ```
   Or with OpenAI: `-e OPENAI_API_KEY=sk-...`

`GET /ready` returns `provider: "gemini"` or `provider: "openai"` when an API key is used; no Ollama check is performed.

---

## Deploy with Docker (Ollama)

### Build and run (single container)

Use this when Ollama is already running on the host (or another reachable URL).

1. Build the image:
   ```bash
   docker build -t ats-pdf-reader .
   ```

2. Run the container (replace with your Ollama URL if different):
   - **Windows/macOS (Ollama on host):**  
     `docker run -p 8000:8000 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 ats-pdf-reader`
   - **Linux (Ollama on host):**  
     `docker run -p 8000:8000 -e OLLAMA_BASE_URL=http://172.17.0.1:11434 ats-pdf-reader`
   - **Ollama at a known URL:**  
     `docker run -p 8000:8000 -e OLLAMA_BASE_URL=http://your-ollama-host:11434 ats-pdf-reader`

3. **Pull the model in Ollama** (required before first extraction). Use the same Ollama instance the API uses:
   - **Ollama on host:** run `ollama pull llama3.2` (or your `MODEL_ID`) on the host.
   - **Ollama in another container:** `docker exec -it <ollama_container_name> ollama pull llama3.2`

4. Open the API: `http://localhost:8000`. Docs: `http://localhost:8000/docs`.

### Deploy with Docker Compose (API + Ollama)

One-command deploy: API and Ollama run in the same Compose stack; the API talks to Ollama over the internal network.

1. From the project directory:
   ```bash
   docker-compose up -d
   ```

2. **Pull an extraction model (required once).** Without this, `POST /extract` returns: `Can't find Ollama llama3.2`.
   ```bash
   docker-compose exec ollama ollama pull llama3.2
   ```
   To use another model (e.g. `mistral`), pull it and set `MODEL_ID` in `.env`:
   ```bash
   docker-compose exec ollama ollama pull mistral
   ```
   Then set `MODEL_ID=mistral` in `.env` and restart: `docker-compose up -d`.

3. API is at `http://localhost:8000`. Optional: set `PORT=9000` in `.env` to expose on 9000.

---

## Deploy without Docker

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Ensure Ollama is running and the chosen model is pulled:
   ```bash
   ollama serve
   ollama pull llama3.2
   ```

3. Run the API (optional: set `OLLAMA_BASE_URL` / `PORT` in the environment):
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```
   Or: `python main.py serve`

4. Open `http://localhost:8000/docs` for the interactive API docs.

---

## Health and readiness

- **`GET /health`** — Liveness: returns `{"status": "ok"}` if the API is up. Use for simple liveness probes.
- **`GET /ready`** — Readiness: when using **Ollama**, returns 200 if Ollama is reachable, 503 otherwise. When using an **API key** (Gemini/OpenAI), returns 200 with `provider` and `api_key_set`. Use for readiness probes and load balancers.

Example:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

In Docker you can add a healthcheck (see `docker-compose.yml`). For Kubernetes, use `/health` for liveness and `/ready` for readiness.

---

## Example: Extract resume JSON

Upload a PDF and receive structured JSON:

```bash
curl -X POST -F "file=@resume.pdf" http://localhost:8000/extract
```

Response shape (see OpenAPI schema at `/docs`): `name`, `email`, `phone`, `summary`, `experience[]`, `education[]`, `skills[]`.

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| **"Extraction failed: Ollama API error: Can't find Ollama llama3.2"** | The model is not pulled in the Ollama instance your API uses. **Fix:** Pull it in that same Ollama. If Ollama runs in Docker: `docker-compose exec ollama ollama pull llama3.2` or `docker exec -it <ollama_container> ollama pull llama3.2`. If Ollama runs locally: `ollama pull llama3.2`. To use a different model: pull it (e.g. `ollama pull mistral`) and set `MODEL_ID=mistral`. |
| **Connection refused to Ollama** | Ollama is not running, or `OLLAMA_BASE_URL` is wrong. From the API host/container, `curl $OLLAMA_BASE_URL/api/tags` should return JSON. For Docker with Ollama on host, use `host.docker.internal` (Windows/Mac) or the host gateway IP on Linux. |
| **Model not found** (other variants) | Same as above: pull the model in the Ollama your API talks to. List existing models: `ollama list` (local) or `docker exec <ollama_container> ollama list`; then set `MODEL_ID` to one of those names or pull the default with `ollama pull llama3.2`. |
| **Invalid PDF / 400** | Ensure the uploaded file is a real PDF and has a `.pdf` extension. Max size is 20 MB. |
| **500 extraction failed** | Check API logs. With Ollama: timeout or model error; try a smaller model. With API key: invalid key, quota, or model name; check `MODEL_ID` and provider docs. |
| **Using API key but still hitting Ollama** | Provider order: `OPENAI_API_KEY` → OpenAI; then `LANGEXTRACT_API_KEY` → Gemini; otherwise Ollama. Ensure the chosen key is set in the same environment the API process uses. |

For more detail on the API and response format, see the project [README](README.md) and the OpenAPI docs at `/docs`.
