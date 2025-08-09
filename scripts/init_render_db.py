#!/usr/bin/env python3
"""
Database initialization script for Render deployment.
This script initializes the database and loads sample data.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from database.connection import init_database, check_database_health
from scripts.sample_data import generate_sample_data


def setup_logging():
    """Configure logging."""
    logger.remove()  # Remove default handler
    logger.add(sys.stdout, level="INFO")
    logger.add("logs/render_init.log", rotation="10 MB", level="DEBUG")


def main():
    """Main initialization function."""
    setup_logging()
    
    logger.info("🏗️ Initializing database for Render deployment...")
    
    # Check if we're running on Render
    is_render = os.environ.get("RENDER", "") == "true"
    if is_render:
        logger.info("Running on Render platform")
    
    # Check database connection
    logger.info("🔍 Checking database connection...")
    if not check_database_health():
        logger.error("❌ Database connection failed. Please check your configuration.")
        sys.exit(1)
    
    # Initialize database tables
    try:
        logger.info("📊 Creating database tables...")
        init_database()
        logger.info("✅ Database tables created successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        sys.exit(1)
    
    # Generate sample data
    try:
        logger.info("🔄 Generating sample data...")
        num_properties = generate_sample_data()
        logger.info(f"✅ Generated {num_properties} sample properties")
    except Exception as e:
        logger.error(f"❌ Sample data generation failed: {e}")
        logger.warning("Continuing without sample data...")
    
    logger.info("🎉 Database initialization completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())