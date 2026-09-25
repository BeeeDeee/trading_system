#!/bin/bash

set -e

echo "======================================="
echo " Trading Platform Status"
echo "======================================="
echo

echo "=== Docker Compose ==="
docker compose -f ~/docker/core/docker-compose.yml ps

echo
echo "=== Docker Containers ==="
docker ps

echo
echo "=== Disk ==="
df -h /

echo
echo "=== Memory ==="
free -h

echo
echo "=== Docker Volumes ==="
docker volume ls

echo
echo "=== Docker Images ==="
docker images