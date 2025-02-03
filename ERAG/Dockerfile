# Use NVIDIA CUDA base image with Python support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Copy all files (Python script + CSV files) into the container
COPY . .

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir \
    pandas \
    torch \
    faiss-cpu \
    langchain \
    langchain-community \
    sentence-transformers \
    requests

# Set the entry point to run your script
CMD ["python3"]

