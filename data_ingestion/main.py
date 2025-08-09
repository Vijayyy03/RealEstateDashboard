#!/usr/bin/env python3
"""
Main data ingestion script for the Real Estate Investment Decision System.
Orchestrates all data collection activities including MLS scraping, tax records, and zoning data.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import argparse
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from config.settings import settings
from database.connection import init_database, check_database_health
from data_ingestion.mls_scraper import MLSScraper
from data_ingestion.tax_scraper import MockTaxRecordsScraper
from data_ingestion.zoning_scraper import MockZoningDataScraper
from data_ingestion.base_scraper import BatchProcessor


def setup_logging():
    """Setup logging configuration."""
    # Create logs directory
    log_dir = Path(settings.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logger
    logger.remove()  # Remove default handler
    logger.add(
        settings.log_file,
        level=settings.log_level,
        rotation="10 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="{time:HH:mm:ss} | {level} | {message}"
    )


def run_mls_scraping():
    """Run MLS data scraping."""
    logger.info("🚀 Starting MLS data scraping...")
    
    try:
        scraper = MLSScraper()
        success = scraper.run()
        
        if success:
            logger.info("✅ MLS scraping completed successfully")
            return True
        else:
            logger.error("❌ MLS scraping failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ MLS scraping error: {e}")
        return False


def run_tax_records_scraping():
    """Run tax records scraping."""
    logger.info("🚀 Starting tax records scraping...")
    
    try:
        scraper = MockTaxRecordsScraper()
        success = scraper.run()
        
        if success:
            logger.info("✅ Tax records scraping completed successfully")
            return True
        else:
            logger.error("❌ Tax records scraping failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Tax records scraping error: {e}")
        return False


def run_zoning_data_scraping():
    """Run zoning data scraping."""
    logger.info("🚀 Starting zoning data scraping...")
    
    try:
        scraper = MockZoningDataScraper()
        success = scraper.run()
        
        if success:
            logger.info("✅ Zoning data scraping completed successfully")
            return True
        else:
            logger.error("❌ Zoning data scraping failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Zoning data scraping error: {e}")
        return False


def run_api_data_collection():
    """Run API-based data collection."""
    logger.info("🚀 Starting API data collection...")
    
    # This would integrate with external APIs like Zillow, Redfin, etc.
    # For now, we'll just log that it's not implemented
    logger.info("⚠️ API data collection not implemented yet")
    return True


def run_underwriting_calculations():
    """Run underwriting calculations for all properties."""
    logger.info("🚀 Starting underwriting calculations...")
    
    try:
        from underwriting.calculator import UnderwritingCalculator
        from database.connection import db_manager
        
        calculator = UnderwritingCalculator()
        
        # Get all property IDs without underwriting data
        with db_manager.get_session() as session:
            from database.models import Property, UnderwritingData
            
            property_ids = session.query(Property.id).outerjoin(
                UnderwritingData, Property.id == UnderwritingData.property_id
            ).filter(
                Property.is_active == True,
                UnderwritingData.id == None
            ).all()
        
        if not property_ids:
            logger.info("No properties need underwriting calculations")
            return True
        
        property_ids = [pid[0] for pid in property_ids]  # Extract IDs from tuples
        logger.info(f"Calculating underwriting for {len(property_ids)} properties")
        
        success_count = 0
        for property_id in property_ids:
            try:
                calculator.calculate_all_metrics(property_id)
                success_count += 1
                logger.info(f"✅ Calculated underwriting for property {property_id}")
            except Exception as e:
                logger.error(f"❌ Failed to calculate underwriting for property {property_id}: {e}")
        
        logger.info(f"✅ Underwriting calculations completed: {success_count}/{len(property_ids)} successful")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"❌ Underwriting calculations error: {e}")
        return False


def run_ml_scoring():
    """Run ML model training and scoring."""
    logger.info("🚀 Starting ML model training and scoring...")
    
    try:
        from ml_models.deal_scorer import DealScorer
        
        scorer = DealScorer()
        
        # Check if we have enough data for training
        X, y = scorer.prepare_training_data()
        
        if len(X) < 10:
            logger.warning("Not enough data for ML model training (need at least 10 samples)")
            return True
        
        # Train model
        logger.info(f"Training model with {len(X)} samples...")
        training_results = scorer.train_model(X, y)
        
        logger.info(f"✅ Model training completed. Test R²: {training_results['test_score']:.3f}")
        
        # Score all properties
        logger.info("Scoring all properties...")
        scoring_results = scorer.score_all_properties()
        
        logger.info(f"✅ ML scoring completed: {len(scoring_results)} properties scored")
        return True
        
    except Exception as e:
        logger.error(f"❌ ML scoring error: {e}")
        return False


def run_full_pipeline():
    """Run the complete data ingestion pipeline."""
    logger.info("🚀 Starting full data ingestion pipeline...")
    
    start_time = datetime.now()
    
    # Check database health
    if not check_database_health():
        logger.error("❌ Database health check failed")
        return False
    
    # Run data collection
    success = True
    
    if settings.scraping.mls_enabled:
        success &= run_mls_scraping()
    
    if settings.scraping.tax_records_enabled:
        success &= run_tax_records_scraping()
    
    if settings.scraping.zoning_enabled:
        success &= run_zoning_data_scraping()
    
    # Run underwriting calculations
    success &= run_underwriting_calculations()
    
    # Run ML scoring
    success &= run_ml_scoring()
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    if success:
        logger.info(f"✅ Full pipeline completed successfully in {duration}")
    else:
        logger.error(f"❌ Pipeline failed after {duration}")
    
    return success


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Real Estate Data Ingestion Pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "mls", "tax", "zoning", "api", "underwriting", "ml"],
        default="full",
        help="Data ingestion mode"
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize database before running"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    
    logger.info("🏗️ Real Estate Investment Data Ingestion Pipeline")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Environment: {settings.environment}")
    
    # Initialize database if requested
    if args.init_db:
        logger.info("Initializing database...")
        try:
            init_database()
            logger.info("✅ Database initialized successfully")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            return 1
    
    # Run based on mode
    try:
        if args.mode == "full":
            success = run_full_pipeline()
        elif args.mode == "mls":
            success = run_mls_scraping()
        elif args.mode == "tax":
            success = run_tax_records_scraping()
        elif args.mode == "zoning":
            success = run_zoning_data_scraping()
        elif args.mode == "api":
            success = run_api_data_collection()
        elif args.mode == "underwriting":
            success = run_underwriting_calculations()
        elif args.mode == "ml":
            success = run_ml_scoring()
        else:
            logger.error(f"Unknown mode: {args.mode}")
            return 1
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
