# ============================================================
# 1) Base Image — Python + CUDA Libraries (for GPU ONNX Runtime)
# ============================================================
# Official python-slim image does NOT include CUDA.
# For GPU, we use Nvidia CUDA runtime first, then install Python.
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Enable fast bfloat16 math, better thread performance
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4

# Avoid .pyc + enable logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ============================================================
# 2) Install system libs + Python
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    libssl-dev \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# 3) Use python3 as default python
# ============================================================
RUN ln -s /usr/bin/python3 /usr/bin/python

# Working directory
WORKDIR /app

# ============================================================
# 4) Copy requirements first (layer caching)
# ============================================================
COPY requirements.txt .

# ============================================================
# 5) Install Python dependencies
# ============================================================
# IMPORTANT: install GPU version of ONNX Runtime
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Replace CPU ORT with GPU-accelerated version
RUN pip uninstall -y onnxruntime || true
RUN pip install --no-cache-dir onnxruntime-gpu==1.18.0

# Install user dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# 6) Copy the full app
# ============================================================
COPY . .

# ============================================================
# 7) Expose FastAPI WebSocket port
# ============================================================
EXPOSE 8000

# ============================================================
# 8) Launch API using uvicorn
# ============================================================
CMD ["uvicorn", "multimodel_api:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--http", "httptools"]
