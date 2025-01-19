# Use a lightweight base image for building
FROM python:3.9-slim AS builder

# Set the working directory
WORKDIR /app

# Install dependencies needed for building
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libssl-dev \
    musl-dev \
    apt-utils && \
    rm -rf /var/lib/apt/lists/*

# Install huggingface_hub and Hugging Face CLI
RUN pip install huggingface_hub

# Create necessary directories for huggingface cache
RUN mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser/.cache

# Install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Use a minimal final image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Create a non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create necessary directories for huggingface cache
RUN mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser/.cache

# Copy necessary parts from the builder stage
COPY --from=builder /usr/local/lib/python3.9 /usr/local/lib/python3.9
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app .

# Set permissions and switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose port 8080
EXPOSE 8080

# Set the entry point for the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]