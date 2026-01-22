FROM python:3.11-slim

WORKDIR /opt/dagster/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml .

# Install the package
RUN pip install --no-cache-dir -e .

# Set Dagster home
ENV DAGSTER_HOME=/opt/dagster/dagster_home
RUN mkdir -p $DAGSTER_HOME

# Expose gRPC port
EXPOSE 4000

# Run Dagster code server
CMD ["dagster", "api", "grpc", "-h", "0.0.0.0", "-p", "4000", "-m", "brev_pipelines.definitions"]