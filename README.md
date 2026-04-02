# ATS PDF Reader

Upload a resume PDF and get structured JSON (name, contact, experience, education, skills) using [LangExtract](https://github.com/google/langextract) with **local [Ollama](https://ollama.com/)** or a **Google Gemini / OpenAI API key**.

## Quick start (local)

1. **Install dependencies** (Python 3.10+):
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

2. **Choose backend:** either run **Ollama** and pull a model, or set an **API key** (no Ollama needed):
   - **Ollama:** `ollama serve` then `ollama pull llama3.2`
   - **Gemini:** set `LANGEXTRACT_API_KEY` (get a key at [Google AI Studio](https://aistudio.google.com/))
   - **OpenAI:** set `OPENAI_API_KEY`

3. **Start the API**:
   ```bash
   uvicorn api:app --reload --port 8000
   ```
   Or: `python main.py serve --reload`

4. **Try it**: open [http://localhost:8000/docs](http://localhost:8000/docs), or:
   ```bash
   curl -X POST -F "file=@resume.pdf" http://localhost:8000/extract
   ```

## Response format (JSON)

The `POST /extract` endpoint returns a JSON object with:

- `name`, `email`, `phone`, `summary`
- `experience`: list of `{ title, company, dates, description }`
- `education`: list of `{ degree, institution, dates }`
- `skills`: list of strings

See the [OpenAPI schema](http://localhost:8000/docs) for the exact Pydantic model.

## CLI (no API)

Extract a PDF file to JSON from the command line:

```bash
python main.py extract path/to/resume.pdf
python main.py extract path/to/resume.pdf --output out.json
```

Requires Ollama running and the model pulled (same as above).

## How to deploy

For production deployment (Docker, Docker Compose, or bare metal), environment variables, health checks, and troubleshooting, see **[DEPLOY.md](DEPLOY.md)**.

## License

Use as you like; dependencies have their own licenses (LangExtract Apache 2.0, etc.).
