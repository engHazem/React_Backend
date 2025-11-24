#!/bin/bash
gunicorn -k uvicorn.workers.UvicornWorker multimodel_api:app --bind=0.0.0.0:8000 --timeout 600
