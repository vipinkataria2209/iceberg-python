#!/bin/bash
# Full Stack Vector Demo Runner

set -e

echo "=========================================="
echo "PyIceberg VectorLake - Full Stack Demo"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "\n${YELLOW}[1/4]${NC} Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker found${NC}"

echo -e "\n${YELLOW}[2/4]${NC} Starting Hive + MinIO with Docker Compose..."
docker compose -f dev/docker-compose-integration.yml up -d

echo -e "${GREEN}✅ Services started${NC}"
echo "   - Hive Metastore: localhost:9083"
echo "   - MinIO: localhost:9000"
echo "   - MinIO Console: http://localhost:9001 (admin/password)"

echo -e "\n${YELLOW}[3/4]${NC} Waiting for services to be ready..."
echo -n "   Waiting 30 seconds"
for i in {1..30}; do
    sleep 1
    echo -n "."
done
echo -e " ${GREEN}Done!${NC}"

# Check if Hive is ready
echo -e "\n${YELLOW}[4/4]${NC} Checking Hive metastore..."
max_retries=10
for i in $(seq 1 $max_retries); do
    if nc -z localhost 9083 2>/dev/null; then
        echo -e "${GREEN}✅ Hive metastore ready${NC}"
        break
    fi
    if [ $i -eq $max_retries ]; then
        echo -e "${RED}❌ Hive metastore not ready after ${max_retries} retries${NC}"
        echo "Check logs: docker compose -f dev/docker-compose-integration.yml logs hive"
        exit 1
    fi
    echo "   Retry $i/$max_retries..."
    sleep 2
done

echo -e "\n${GREEN}=========================================="
echo "🚀 Running Vector Search Demo"
echo "==========================================${NC}\n"

# Run the demo
python demo_vector_full_stack.py

echo -e "\n${YELLOW}=========================================="
echo "To stop services:"
echo "==========================================${NC}"
echo "docker compose -f dev/docker-compose-integration.yml down -v"
echo ""
echo "To view logs:"
echo "docker compose -f dev/docker-compose-integration.yml logs -f"

