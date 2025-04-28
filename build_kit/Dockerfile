# Use a base image that supports multiple languages, e.g., Debian
FROM ghcr.io/make87/debian:bookworm AS base

# Set a working directory
WORKDIR /app

# Install common dependencies (adjust as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files into the container (adjust if needed)
COPY . .

# Uncomment and modify the following lines based on your project:
#
# Install language-specific dependencies
# RUN apt-get install -y python3 python3-pip python3-venv libpython3-dev \  # For Python
#     && python3 -m venv /make87/venv \
#     && /make87/venv/bin/pip install --upgrade pip setuptools wheel \
#     && /make87/venv/bin/pip install -r requirements.txt \
#     && /make87/venv/bin/pip install .  # Ensure setup.py or pyproject.toml is installed
#
#
# Alternatively, use language-specific package managers
# RUN cargo build --release  # For Rust

# Define the entry point (adjust based on project type)
# ENTRYPOINT ["/app/main"]  # Adjust as needed for Rust, Go, or other compiled binaries
# ENTRYPOINT ["/make87/venv/bin/python3", "-m", "app.main"]  # Python example

# Expose ports if needed (adjust based on application needs)
# EXPOSE 8080  # Common for web apps

# Default command, adjust as needed
CMD ["/app/main"]
