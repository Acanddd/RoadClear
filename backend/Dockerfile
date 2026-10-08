FROM python:3.12.10-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 ROADCLEAR_DEVICE=cpu ROADCLEAR_MODEL_DIR=/app/models ROADCLEAR_DATA_DIR=/app/runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.lock.txt /app/backend/requirements.lock.txt
RUN pip install -r /app/backend/requirements.lock.txt
COPY backend/app /app/backend/app
COPY scripts /app/scripts
COPY samples /app/samples
COPY docs/model-manifest.json /app/docs/model-manifest.json
COPY frontend/image_display /app/frontend/image_display
COPY frontend/vue/public/algorithm-showcase /app/frontend/vue/public/algorithm-showcase
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--app-dir", "/app/backend", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
