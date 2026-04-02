# ATS PDF Reader API — production image
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api.py pdf_reader.py resume_extract.py schemas.py ./

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Override HOST/PORT/OLLAMA_BASE_URL via env when running
ENV HOST=0.0.0.0 PORT=8000
CMD ["sh", "-c", "exec python -m uvicorn api:app --host $HOST --port $PORT"]
