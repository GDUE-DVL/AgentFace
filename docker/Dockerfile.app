# AgentFace Main Application Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies for Pillow and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code
COPY src/ ./src/

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "agent_face.main:app", "--host", "0.0.0.0", "--port", "8000"]
