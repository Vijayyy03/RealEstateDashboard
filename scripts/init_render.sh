#!/bin/bash

# Initialize database for Render deployment
echo "Starting database initialization for Render..."

# Create logs directory if it doesn't exist
mkdir -p logs

# Run database initialization script
python scripts/init_render_db.py

echo "Database initialization completed."