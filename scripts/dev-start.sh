#!/bin/bash
# =============================================================================
# Start local development environment for Dagster pipelines
# =============================================================================
# Mirrors production K8s configuration from k8s/apps/
# =============================================================================
set -e

cd "$(dirname "$0")/.."

# Use docker-compose or docker compose depending on what's available
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
elif docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    echo "ERROR: Neither docker-compose nor docker compose found"
    exit 1
fi

echo "==================================================================="
echo "  Dagster Local Development Environment"
echo "  (Mirrors production K8s configuration)"
echo "==================================================================="
echo ""
echo "Services:"
echo "  - MinIO     localhost:9000  (console: localhost:9001)"
echo "  - LakeFS    localhost:8000"
echo "  - Weaviate  localhost:8080  (gRPC: localhost:50051)"
echo ""
echo "Configuration mirrors:"
echo "  - k8s/apps/minio/values.yaml"
echo "  - k8s/apps/lakefs/values.yaml"
echo "  - k8s/apps/weaviate/values.yaml"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

# Start services
echo "Starting Docker services..."
$DOCKER_COMPOSE -f docker-compose.dev.yml up -d

# Wait for services to be healthy
echo ""
echo "Waiting for services to initialize..."
sleep 8

# Check service health
echo ""
echo "Service Status:"
echo "-------------------------------------------------------------------"

# MinIO
if curl -sf http://localhost:9000/minio/health/live > /dev/null 2>&1; then
    echo "  MinIO:    ✓ Ready"
    echo "            API:     http://localhost:9000"
    echo "            Console: http://localhost:9001 (admin/password123)"
else
    echo "  MinIO:    ✗ Not ready yet"
fi

echo ""

# LakeFS
if curl -sf http://localhost:8000/api/v1/healthcheck > /dev/null 2>&1; then
    echo "  LakeFS:   ✓ Ready"
    echo "            UI:      http://localhost:8000"
    echo "            Repo:    data (branches: main, staging)"
    echo "            Access:  AKIAIOSFOLKFSSAMPLES"
else
    echo "  LakeFS:   ✗ Not ready yet"
fi

echo ""

# Weaviate
if curl -sf http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; then
    echo "  Weaviate: ✓ Ready"
    echo "            HTTP:    http://localhost:8080"
    echo "            gRPC:    localhost:50051"
else
    echo "  Weaviate: ✗ Not ready yet"
fi

echo ""
echo "-------------------------------------------------------------------"
echo ""
echo "LLM Services (mock fallback - no GPU needed):"
echo "  - NIM Embedding: Returns deterministic hash-based vectors"
echo "  - NIM LLM:       Returns neutral fallback values"
echo "  - Tracking:      _llm_status, _llm_fallback_used columns"
echo ""
echo "-------------------------------------------------------------------"
echo ""
echo "To start Dagster:"
echo "  uv run dagster dev -m brev_pipelines.definitions"
echo ""
echo "Then open: http://localhost:3000"
echo ""
echo "Useful commands:"
echo "  $DOCKER_COMPOSE -f docker-compose.dev.yml logs -f     # View logs"
echo "  $DOCKER_COMPOSE -f docker-compose.dev.yml down        # Stop services"
echo "  $DOCKER_COMPOSE -f docker-compose.dev.yml down -v     # Stop + remove data"
echo "==================================================================="
