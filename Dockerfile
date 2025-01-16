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
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Use a minimal final image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy only necessary parts from the builder stage
COPY --from=builder /usr/local/lib/python3.9 /usr/local/lib/python3.9
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app .

# Expose port 8000
EXPOSE 8000

# Set the entry point for the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]