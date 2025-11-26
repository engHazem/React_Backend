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
# 3) Set working directory
# ============================================================
WORKDIR /app

# ============================================================
# 4) Install Python dependencies
# ============================================================
COPY requirements.txt .

# Upgrade pip + build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# FIX NumPy problem (force version < 2)
RUN pip install --no-cache-dir "numpy<2"

# Install ONNX Runtime CPU (compatible with numpy<2)
RUN pip install --no-cache-dir onnxruntime==1.18.0

# Install your other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# 5) Copy application
# ============================================================
COPY . .

# ============================================================
# 6) Expose FastAPI port (Railway uses PORT variable)
# ============================================================
EXPOSE 8000

# ============================================================
# 7) Run FastAPI using Uvicorn
# ============================================================
CMD ["uvicorn", "multimodel_api:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--http", "httptools"]
