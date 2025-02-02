# Stage 1: Build dependencies
FROM python:3.9-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies for compiling Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies in one step
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "pydantic<2.0.0" spacy -r requirements.txt \
    && python -m spacy download en_core_web_sm \
    && rm -rf ~/.cache/pip

# Copy application source files
COPY . .


# Stage 2: Create minimal final image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser

# Copy only necessary files from builder
# If data is required
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app
COPY ./data /app/data 
#

# Ensure correct permissions and switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose the app port
EXPOSE 8080

# Run the app securely
CMD ["python", "-m", "app"]