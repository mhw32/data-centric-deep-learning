# Use Python 3.10 base image
FROM mcr.microsoft.com/devcontainers/python:3.10

# Set environment variables
ENV PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    redis-server \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /dcdl_projects

# Copy and install Python requirements
COPY requirements.txt /dcdl_projects/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /dcdl_projects/

# Set entrypoint
CMD ["bash"]