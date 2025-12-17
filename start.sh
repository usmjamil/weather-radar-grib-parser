#!/bin/bash
set -e

echo "Starting Python GRIB2 Parser Service..."
echo "Python version: $(python --version)"
echo "Installing dependencies..."
pip install --no-cache-dir -r requirements.txt
echo "Dependencies installed. Starting server..."
exec gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT:-8000}
