#!/bin/bash
set -euo pipefail

echo "Waiting for database to be ready..."

DB_USER=${POSTGRES_USER:-hexis_user}
DB_NAME=${POSTGRES_DB:-hexis_memory}

# Run inside the container, so call postgres utilities directly (use the local socket)
until pg_isready -U "${DB_USER}" -d "${DB_NAME}" > /dev/null 2>&1; do
    echo "Database is not ready yet, waiting..."
    sleep 2
done

echo "Database is ready!"

echo "Testing database connection and extensions..."
if psql -U "${DB_USER}" -d "${DB_NAME}" -c "SELECT 1;" > /dev/null 2>&1; then
    echo "Database is fully operational!"
else
    echo "Database connection test failed"
    exit 1
fi
