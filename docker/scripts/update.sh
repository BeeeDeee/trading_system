#!/bin/bash

set -e

COMPOSE_DIR="$HOME/docker/core"
COMPOSE_FILE="$COMPOSE_DIR/docker-compose.yml"

echo "======================================="
echo " Trading Platform Update"
echo "======================================="
echo

# Docker service check
if ! systemctl is-active --quiet docker; then
    echo "Docker service is not running!"
    exit 1
fi

echo "[1/5] Creating backup..."
"$HOME/docker/scripts/backup.sh"

echo
echo "[2/5] Pulling new images..."
docker compose -f "$COMPOSE_FILE" pull

echo
echo "[3/5] Restarting containers..."
docker compose -f "$COMPOSE_FILE" up -d

echo
echo "[4/5] Waiting for services..."
sleep 10

echo
echo "[5/5] Current status:"
docker compose -f "$COMPOSE_FILE" ps

echo
echo "Update finished successfully."