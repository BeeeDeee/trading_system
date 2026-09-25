#!/bin/bash

# Check if the PostgreSQL container is running
docker ps --format "{{.Names}}" | grep -q "^postgres$" || {
    echo "PostgreSQL container is not running."
    exit 1
}

set -e

# Load environment variables from .env file
set -a
source ~/docker/core/.env
set +a

# Create backup directory if it doesn't exist
DATE=$(date +"%Y-%m-%d_%H-%M-%S")
BACKUP_DIR="$HOME/docker/backups"

mkdir -p "$BACKUP_DIR"

echo "Creating PostgreSQL backup..."

# Run pg_dump inside the PostgreSQL container and compress the output
docker exec postgres \
    pg_dump \
    -U "$POSTGRES_USER" \
    "$POSTGRES_DB" \
| gzip > "$BACKUP_DIR/postgres_$DATE.sql.gz"

echo
echo "Backup created:"
echo "$BACKUP_DIR/postgres_$DATE.sql.gz"

echo
echo "Removing backups older than 14 days..."

# Find and delete backup files older than 14 days
find "$BACKUP_DIR" \
    -name "*.sql.gz" \
    -type f \
    -mtime +14 \
    -delete

echo
ls -lh "$BACKUP_DIR"

