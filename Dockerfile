FROM python:3.11-slim

WORKDIR /app

# System deps for pypdfium2 and pillow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install chandra + FastAPI deps
COPY pyproject.toml .
COPY chandra/ chandra/
RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir fastapi uvicorn python-multipart

# Default env
ENV VLLM_API_BASE=http://host.docker.internal:8000/v1
ENV VLLM_MODEL_NAME=chandra
ENV VLLM_API_KEY=EMPTY

EXPOSE 8080

CMD ["uvicorn", "chandra.scripts.api:app", "--host", "0.0.0.0", "--port", "8080"]
