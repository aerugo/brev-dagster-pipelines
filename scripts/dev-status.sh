#!/usr/bin/env bash
# =============================================================================
# Status of local development environment for Dagster pipelines
# =============================================================================
# Shows health, memory usage, and connectivity of all dev services
# =============================================================================

cd "$(dirname "$0")/.."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Symbols
CHECK="${GREEN}✓${NC}"
CROSS="${RED}✗${NC}"
WARN="${YELLOW}⚠${NC}"

# Container name to display name mapping
get_display_name() {
    case "$1" in
        dagster-minio) echo "MinIO" ;;
        dagster-lakefs) echo "LakeFS" ;;
        dagster-weaviate) echo "Weaviate" ;;
        *) echo "$1" ;;
    esac
}

echo ""
echo -e "${BOLD}==================================================================="
echo "  Dagster Development Environment Status"
echo -e "===================================================================${NC}"

# -----------------------------------------------------------------------------
# Docker/Colima Status
# -----------------------------------------------------------------------------
echo ""
echo -e "${BOLD}Docker Runtime:${NC}"
echo "-------------------------------------------------------------------"

if ! docker info > /dev/null 2>&1; then
    echo -e "  ${CROSS} Docker is not running"
    echo ""
    echo "  Start with: colima start --memory 6 --cpu 4"
    exit 1
fi

# Check if using Colima
if command -v colima &> /dev/null && colima status &> /dev/null; then
    COLIMA_INFO=$(colima list 2>/dev/null | grep -E "^default|Running" | head -1)
    if [ -n "$COLIMA_INFO" ]; then
        CPUS=$(echo "$COLIMA_INFO" | awk '{print $4}')
        MEMORY=$(echo "$COLIMA_INFO" | awk '{print $5}')
        echo -e "  ${CHECK} Colima running"
        echo -e "     CPUs:   ${CYAN}${CPUS}${NC}"
        echo -e "     Memory: ${CYAN}${MEMORY}${NC}"

        # Warn if memory is low
        MEM_GB=$(echo "$MEMORY" | sed 's/GiB//')
        if (( $(echo "$MEM_GB < 4" | bc -l) )); then
            echo -e "     ${WARN} ${YELLOW}Low memory - recommend 4GB+ for Weaviate${NC}"
            echo "        Run: colima stop && colima start --memory 6"
        fi
    fi
else
    DOCKER_MEM=$(docker info --format '{{.MemTotal}}' 2>/dev/null)
    if [ -n "$DOCKER_MEM" ]; then
        MEM_GB=$(echo "scale=1; $DOCKER_MEM / 1024 / 1024 / 1024" | bc)
        echo -e "  ${CHECK} Docker running (${MEM_GB}GB memory)"
    fi
fi

# -----------------------------------------------------------------------------
# Container Status
# -----------------------------------------------------------------------------
echo ""
echo -e "${BOLD}Containers:${NC}"
echo "-------------------------------------------------------------------"

for container in "dagster-minio" "dagster-lakefs" "dagster-weaviate"; do
    name=$(get_display_name "$container")

    # Check if container exists and get status
    STATUS=$(docker inspect --format='{{.State.Status}}' "$container" 2>/dev/null)
    HEALTH=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null)

    if [ -z "$STATUS" ]; then
        printf "  %-10s ${CROSS} Not created\n" "$name:"
    elif [ "$STATUS" != "running" ]; then
        printf "  %-10s ${CROSS} ${RED}%s${NC}\n" "$name:" "$STATUS"
    elif [ "$HEALTH" = "healthy" ]; then
        # Get memory usage
        MEM=$(docker stats --no-stream --format "{{.MemUsage}}" "$container" 2>/dev/null | cut -d'/' -f1 | xargs)
        printf "  %-10s ${CHECK} Running (${CYAN}%s${NC})\n" "$name:" "$MEM"
    elif [ "$HEALTH" = "unhealthy" ]; then
        printf "  %-10s ${WARN} ${YELLOW}Unhealthy${NC}\n" "$name:"
    else
        MEM=$(docker stats --no-stream --format "{{.MemUsage}}" "$container" 2>/dev/null | cut -d'/' -f1 | xargs)
        printf "  %-10s ${CHECK} Running (${CYAN}%s${NC}) - no healthcheck\n" "$name:" "$MEM"
    fi
done

# -----------------------------------------------------------------------------
# Service Health Checks
# -----------------------------------------------------------------------------
echo ""
echo -e "${BOLD}Service Health:${NC}"
echo "-------------------------------------------------------------------"

# MinIO
if curl -sf http://localhost:9000/minio/health/live > /dev/null 2>&1; then
    echo -e "  MinIO     ${CHECK} http://localhost:9000"
    echo -e "            Console: http://localhost:9001 (admin/password123)"
else
    echo -e "  MinIO     ${CROSS} Not responding on port 9000"
fi

# LakeFS
if curl -sf http://localhost:8000/api/v1/healthcheck > /dev/null 2>&1; then
    # Get repo info
    REPOS=$(curl -s http://localhost:8000/api/v1/repositories \
        -u "AKIAIOSFOLKFSSAMPLES:wJalrXUtnFEMI/K7MDENG/bPxRfiCYSAMPLEKEY" 2>/dev/null \
        | python3 -c "import sys,json; r=json.load(sys.stdin).get('results',[]); print(','.join([x['id'] for x in r]))" 2>/dev/null)
    echo -e "  LakeFS    ${CHECK} http://localhost:8000"
    if [ -n "$REPOS" ]; then
        echo -e "            Repos: ${CYAN}${REPOS}${NC}"
    fi
else
    echo -e "  LakeFS    ${CROSS} Not responding on port 8000"
fi

# Weaviate
if curl -sf http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; then
    # Get collection count
    COLLECTIONS=$(curl -s http://localhost:8080/v1/schema 2>/dev/null \
        | python3 -c "import sys,json; c=json.load(sys.stdin).get('classes',[]); print(len(c))" 2>/dev/null)
    echo -e "  Weaviate  ${CHECK} http://localhost:8080 (gRPC: 50051)"
    if [ -n "$COLLECTIONS" ]; then
        echo -e "            Collections: ${CYAN}${COLLECTIONS}${NC}"
    fi
else
    echo -e "  Weaviate  ${CROSS} Not responding on port 8080"
fi

# -----------------------------------------------------------------------------
# Dagster Status
# -----------------------------------------------------------------------------
echo ""
echo -e "${BOLD}Dagster:${NC}"
echo "-------------------------------------------------------------------"

if curl -sf http://localhost:3000/graphql -H "Content-Type: application/json" -d '{"query":"{__typename}"}' > /dev/null 2>&1; then
    # Get active runs
    RUNS=$(curl -s http://localhost:3000/graphql -H "Content-Type: application/json" \
        -d '{"query":"{ runsOrError { ... on Runs { results { status } } } }"}' 2>/dev/null \
        | python3 -c "import sys,json; r=json.load(sys.stdin)['data']['runsOrError'].get('results',[]); s=[x['status'] for x in r[:20]]; print(f\"Running: {s.count('STARTED')}, Queued: {s.count('QUEUED')}, Failed: {s.count('FAILURE')}\")" 2>/dev/null)
    echo -e "  Webserver ${CHECK} http://localhost:3000"
    if [ -n "$RUNS" ]; then
        echo -e "            ${RUNS}"
    fi
else
    echo -e "  Webserver ${CROSS} Not running"
    echo -e "            Start with: ${CYAN}uv run dagster dev -m brev_pipelines.definitions${NC}"
fi

# -----------------------------------------------------------------------------
# Quick Commands
# -----------------------------------------------------------------------------
echo ""
echo -e "${BOLD}Commands:${NC}"
echo "-------------------------------------------------------------------"
echo "  View logs:    docker-compose -f docker-compose.dev.yml logs -f"
echo "  Restart:      docker-compose -f docker-compose.dev.yml restart"
echo "  Stop:         docker-compose -f docker-compose.dev.yml down"
echo "  Start Dagster: uv run dagster dev -m brev_pipelines.definitions"
echo "==================================================================="
echo ""
