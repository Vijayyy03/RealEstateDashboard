#!/usr/bin/env python3
"""
Database setup script for the Real Estate Investment Decision System.
This script initializes the database, creates tables, and sets up any required extensions.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.connection import init_database, check_database_health
from config.settings import settings
from loguru import logger


def setup_postgis_extension():
    """Setup PostGIS extension if using PostgreSQL."""
    if settings.database.host != "localhost" or "postgresql" in settings.database.url:
        try:
            from database.connection import db_manager
            with db_manager.get_session() as session:
                # Enable PostGIS extension
                session.execute("CREATE EXTENSION IF NOT EXISTS postgis")
                session.execute("CREATE EXTENSION IF NOT EXISTS postgis_topology")
                logger.info("✅ PostGIS extensions enabled")
        except Exception as e:
            logger.warning(f"⚠️ Could not enable PostGIS extensions: {e}")
            logger.info("Continuing without PostGIS support...")


def create_logs_directory():
    """Create logs directory if it doesn't exist."""
    log_dir = Path(settings.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Logs directory created: {log_dir}")


def main():
    """Main setup function."""
    logger.info("🏗️ Starting database setup for Real Estate Investment System...")
    
    try:
        # Create logs directory
        create_logs_directory()
        
        # Check database connection
        logger.info("🔍 Checking database connection...")
        if not check_database_health():
            logger.error("❌ Database connection failed. Please check your configuration.")
            sys.exit(1)
        
        # Setup PostGIS extensions
        logger.info("🗺️ Setting up PostGIS extensions...")
        setup_postgis_extension()
        
        # Initialize database tables
        logger.info("📊 Creating database tables...")
        init_database()
        
        logger.info("🎉 Database setup completed successfully!")
        logger.info(f"📝 Database URL: {settings.database.url}")
        logger.info(f"🏠 Database Name: {settings.database.name}")
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
