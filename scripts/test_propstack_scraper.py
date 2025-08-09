#!/usr/bin/env python3
"""
Test script for the Propstack API scraper.
This script tests the PropstackScraper implementation with demo data.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from data_ingestion.propstack_scraper import PropstackScraper
from config.settings import settings

# Initialize database connection
from database.connection import init_database, db_manager


def setup_logging():
    """Setup logging configuration."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        level="INFO",
        format="{time:HH:mm:ss} | {level} | {message}"
    )
    logger.add("logs/propstack_test.log", rotation="10 MB", level="DEBUG")


def init_db():
    """Initialize the database connection."""
    logger.info("Initializing database connection...")
    init_database()
    logger.info("Database connection initialized")


def test_propstack_scraper():
    """Test the PropstackScraper implementation."""
    logger.info("Testing PropstackScraper with demo data...")
    
    # Initialize the scraper
    scraper = PropstackScraper()
    
    # Test scraping data
    logger.info("Testing scrape_data method...")
    raw_data = scraper.scrape_data(city="Mumbai", property_type="Residential", limit=10)
    
    if not raw_data:
        logger.error("Failed to scrape data")
        return
        
    logger.info(f"Retrieved {len(raw_data)} properties from demo data")
    
    # Test processing data
    logger.info("Testing process_data method...")
    processed_data = scraper.process_data(raw_data)
    
    if not processed_data:
        logger.error("Failed to process data")
        return
        
    logger.info(f"Processed {len(processed_data)} properties")
    
    # Print sample data
    if processed_data:
        sample = processed_data[0]
        logger.info("Sample processed property:")
        for key, value in sample.items():
            logger.info(f"  {key}: {value}")
    
    # Test save_data method
    logger.info("Testing save_data method...")
    try:
        scraper.save_data(processed_data)
        logger.info("Successfully saved data to database")
    except Exception as e:
        logger.error(f"Failed to save data: {e}")
    
    logger.info("✅ PropstackScraper test completed")


def main():
    """Main function."""
    setup_logging()
    init_db()
    logger.info("🏗️ PropstackScraper Test Script")
    
    test_propstack_scraper()
    
    return 0


if __name__ == "__main__":
    exit(main())