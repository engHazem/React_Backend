# ============================================================
# 1) Base Image — Lightweight Python for fast CPU performance
# ============================================================
FROM python:3.10-slim

# Enable optimized math performance
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4

# Avoid .pyc + enable logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ============================================================
# 2) Install system dependencies (for cv2, numpy, mediapipe)
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# 3) Working directory
# ============================================================
WORKDIR /app

# ============================================================
# 4) Install requirements
# ============================================================
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Force CPU ONNX Runtime (your requirements now contain onnxruntime, not GPU)
RUN pip uninstall -y onnxruntime-gpu || true
RUN pip install --no-cache-dir onnxruntime==1.18.0

# Install user dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# 5) Copy app code
# ============================================================
COPY . .

# ============================================================
# 6) Expose FastAPI port
# ============================================================
EXPOSE 8000

# ============================================================
# 7) Run FastAPI using uvicorn + uvloop + httptools
# ============================================================
CMD ["uvicorn", "multimodel_api:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--http", "httptools"]
