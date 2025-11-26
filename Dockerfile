# ============================================================
# 1) Base Image — Lightweight & Fast for CPU
# ============================================================
FROM python:3.10-slim

# Performance optimizations
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4

# Avoid .pyc + enable logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ============================================================
# 2) Install system dependencies (Railway-compatible)
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libglx-mesa0 \
    libgl1-mesa-dri \
    libsm6 \
    libxext6 \
    libxrender1 \
    libssl-dev \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# 3) Working directory
# ============================================================
WORKDIR /app

# Copy requirements before installing
COPY requirements.txt .

# ============================================================
# 4) Install Python dependencies (pinned & conflict-free)
# ============================================================

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ---- CRITICAL: Install pinned NumPy + ONNXRuntime first ----
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir onnxruntime==1.18.0

# ---- Install pinned CV/ML libs WITHOUT upgrading NumPy ----
RUN pip install --no-cache-dir --no-deps \
    opencv-python-headless==4.9.0.80 \
    mediapipe==0.10.9 \
    scikit-learn==1.3.2

# ---- Install the rest from requirements.txt ----
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# 5) Copy application code
# ============================================================
COPY . .

# ============================================================
# 6) Expose FastAPI port
# ============================================================
EXPOSE 8000

# ============================================================
# 7) Run FastAPI using Uvicorn (high-performance)
# ============================================================
CMD ["uvicorn", "multimodel_api:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--http", "httptools"]
