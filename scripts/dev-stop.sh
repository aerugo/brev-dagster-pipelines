#!/bin/bash
# =============================================================================
# Stop local development environment
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

echo "Stopping Docker services..."
$DOCKER_COMPOSE -f docker-compose.dev.yml down

echo ""
echo "Services stopped."
echo ""
echo "To remove volumes (all data):"
echo "  $DOCKER_COMPOSE -f docker-compose.dev.yml down -v"
