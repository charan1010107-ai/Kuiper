FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (libgomp1 for LightGBM, build tools, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight CPU-only PyTorch first to prevent timeouts & memory issues
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application source code
COPY backend/ .

# Render and cloud platforms inject PORT dynamically
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
