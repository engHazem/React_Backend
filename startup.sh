#!/bin/bash
set -e
LISTEN_PORT=${PORT:-8080}

echo "=========================================="
echo "🚀 Starting AI Backend API"
echo "=========================================="

# Print Python info
echo "🐍 Python version:"
python --version
echo "📁 Working directory: $(pwd)"

# Show models directory
echo "📦 Checking model files..."
if [ -d "models" ]; then
    echo "✓ Models directory found:"
    ls -lh models/
else
    echo "⚠️ Warning: models directory not found!"
fi

# Install minimal system deps for OpenCV + MediaPipe
echo "📦 Installing system dependencies..."
apt-get update -y >/dev/null 2>&1 || true
apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 >/dev/null 2>&1 || true

echo "✓ System dependencies installed"

# Verify Python packages
echo "🔍 Verifying Python packages..."
python -c "import fastapi; print('✓ FastAPI installed')" || echo "⚠️ FastAPI not found"
python -c "import cv2; print('✓ OpenCV installed')" || echo "⚠️ OpenCV not found"
python -c "import onnxruntime; print('✓ ONNX Runtime installed')" || echo "⚠️ ONNX Runtime not found"
python -c "import uvloop; print('✓ uvloop installed')" || echo "⚠️ uvloop not found"
python -c "import httptools; print('✓ httptools installed')" || echo "⚠️ httptools not found"

echo "=========================================="
echo "🌐 Starting Uvicorn server (optimized for Azure WebSockets)"
echo "=========================================="

exec uvicorn multimodel_api:app \
  --host 0.0.0.0 \
  --port ${LISTEN_PORT} \
  --workers 1 \
  --loop uvloop \
  --http httptools \
  --timeout-keep-alive 45
