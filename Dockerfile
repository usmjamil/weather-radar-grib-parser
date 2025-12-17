FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directory
RUN mkdir -p /app/cache

# Set default environment variables
ENV SAMPLE_RATE=10
ENV CHUNK_SIZE=1000

EXPOSE 8000

# Start the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
