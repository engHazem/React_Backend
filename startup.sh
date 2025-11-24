#!/bin/bash

apt-get update
apt-get install -y libgl1

python -m uvicorn multimodel_api:app --host 0.0.0.0 --port 8000
