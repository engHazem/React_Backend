
#!/bin/bash
set -e
LISTEN_PORT=${PORT:-8080}
echo "=========================================="
echo "🚀 Starting AI Backend API"
echo "=========================================="

# Print Python version
echo "🐍 Python version:"
python --version

# Print working directory
echo "📁 Working directory:"
pwd

# List files to verify models are present
echo "📦 Checking model files..."
if [ -d "models" ]; then
    echo "✓ Models directory found:"
    ls -lh models/
else
    echo "⚠️  Warning: models directory not found!"
fi

# Install system dependencies for OpenCV (Azure Linux environment)
echo "📦 Installing system dependencies..."
apt-get update > /dev/null 2>&1 || true
apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev > /dev/null 2>&1 || true

echo "✓ System dependencies installed"

# Verify critical Python packages
echo "🔍 Verifying Python packages..."
python -c "import fastapi; print('✓ FastAPI installed')" || echo "⚠️  FastAPI not found"
python -c "import torch; print('✓ PyTorch installed')" || echo "⚠️  PyTorch not found"
python -c "import cv2; print('✓ OpenCV installed')" || echo "⚠️  OpenCV not found"

# Start the application
echo "=========================================="
echo "🌐 Starting Gunicorn server..."
echo "=========================================="

exec gunicorn multimodel_api:app \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:${LISTEN_PORT} \
  --timeout 600 \
  --log-level info \
  --access-logfile - \
  --error-logfile - \
  --capture-output \
  --enable-stdio-inheritance