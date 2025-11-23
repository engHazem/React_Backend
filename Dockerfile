# Use lightweight Python image
FROM python:3.11-slim

# Prevent buffer issues
ENV PYTHONUNBUFFERED=1

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Run FastAPI using Uvicorn
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "multimodel_api:app", "--bind", "0.0.0.0:8000"]
