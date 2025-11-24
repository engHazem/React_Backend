#!/bin/bash
chmod +x startup.sh

# Install system dependencies
apt-get update
apt-get install -y libgl1

# Start FastAPI using Gunicorn + Uvicorn Worker
gunicorn multimodel_api:app \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind=0.0.0.0:8000 \
  --timeout 600
