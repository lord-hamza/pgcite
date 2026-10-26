FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_CACHE_DIR=/models

WORKDIR /app

COPY pyproject.toml README.md ./
COPY pgcite ./pgcite
RUN pip install . && python -m pgcite.models

# Models are baked into the image, so the running service never needs the Hugging Face Hub.
ENV HF_HUB_OFFLINE=1

COPY scripts ./scripts
COPY data ./data

RUN useradd --create-home --uid 1000 app && chown -R app:app /models
USER app

EXPOSE 8000
CMD ["uvicorn", "pgcite.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
