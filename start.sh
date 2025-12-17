#!/bin/bash
set -e

echo "Starting Python GRIB2 Parser Service..."
echo "Python version: $(python --version)"
echo "Memory-optimized configuration for 512MB servers"
echo "Installing dependencies..."
pip install --no-cache-dir -r requirements.txt
echo "Dependencies installed. Starting server..."
# Set memory-optimized environment variables
export SAMPLE_RATE=${SAMPLE_RATE:-20}
export CHUNK_SIZE=${CHUNK_SIZE:-500}
export MAX_POINTS=${MAX_POINTS:-5000}
echo "Using SAMPLE_RATE=$SAMPLE_RATE, CHUNK_SIZE=$CHUNK_SIZE, MAX_POINTS=$MAX_POINTS"
exec gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT:-8000} --max-requests 1000 --max-requests-jitter 100 --timeout 120
