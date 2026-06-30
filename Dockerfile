# Builder stage
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    proj-bin \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt
RUN pip install --user --no-cache-dir gunicorn
RUN pip install --user --no-cache-dir sentence-transformers

# Runner stage
FROM python:3.10-slim AS runner

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    proj-bin \
    libgeos-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

WORKDIR /workspace

# Copy installed python dependencies
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/workspace

# Set cache location for Hugging Face models
ENV HF_HOME=/home/appuser/.cache/huggingface
RUN mkdir -p $HF_HOME && chown -R appuser:appuser $HF_HOME

# Download model during build step (as appuser to ensure permissions)
USER appuser
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application files (switch back to root temporarily if needed, but COPY --chown=appuser handles it)
# We stay as appuser for the rest of the build
COPY --chown=appuser:appuser . .

EXPOSE 8000

# Start production server with Gunicorn and Uvicorn workers
CMD ["gunicorn", "app.api.main:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
