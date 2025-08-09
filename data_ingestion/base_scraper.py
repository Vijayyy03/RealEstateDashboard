"""
Base scraper class for data ingestion components.
Provides common functionality for web scraping and data processing.
"""

import time
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from loguru import logger
from sqlalchemy.orm import Session

from config.settings import settings
from database.connection import db_manager
from database.models import DataIngestionLog


class BaseScraper(ABC):
    """Base class for all data scrapers."""
    
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.session = self._create_session()
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None
        }
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic and headers."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=settings.scraping.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers
        session.headers.update({
            'User-Agent': settings.scraping.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        return session
    
    def _delay(self):
        """Add random delay between requests to be respectful."""
        delay = settings.scraping.request_delay + random.uniform(0, 1)
        time.sleep(delay)
    
    def _make_request(self, url: str, **kwargs) -> Optional[requests.Response]:
        """Make an HTTP request with error handling and logging."""
        try:
            self._delay()
            response = self.session.get(url, timeout=settings.scraping.timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
    
    def _log_ingestion(self, ingestion_type: str, status: str, error_message: str = None):
        """Log ingestion activity to database."""
        try:
            with db_manager.get_session() as session:
                log_entry = DataIngestionLog(
                    source=self.source_name,
                    ingestion_type=ingestion_type,
                    records_processed=self.stats['processed'],
                    records_successful=self.stats['successful'],
                    records_failed=self.stats['failed'],
                    start_time=self.stats['start_time'],
                    end_time=self.stats['end_time'],
                    duration_seconds=(self.stats['end_time'] - self.stats['start_time']).total_seconds() if self.stats['end_time'] else None,
                    status=status,
                    error_message=error_message
                )
                session.add(log_entry)
                session.commit()
        except Exception as e:
            logger.error(f"Failed to log ingestion: {e}")
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate scraped data for required fields."""
        required_fields = self.get_required_fields()
        for field in required_fields:
            if field not in data or data[field] is None:
                logger.warning(f"Missing required field: {field}")
                return False
        return True
    
    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Return list of required fields for this scraper."""
        pass
    
    @abstractmethod
    def scrape_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Scrape data from the source."""
        pass
    
    @abstractmethod
    def process_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean raw scraped data."""
        pass
    
    @abstractmethod
    def save_data(self, processed_data: List[Dict[str, Any]]) -> bool:
        """Save processed data to database."""
        pass
    
    def run(self, **kwargs) -> bool:
        """Main execution method for the scraper."""
        logger.info(f"🚀 Starting {self.source_name} scraper...")
        
        self.stats['start_time'] = datetime.utcnow()
        
        try:
            # Scrape raw data
            logger.info("📥 Scraping raw data...")
            raw_data = self.scrape_data(**kwargs)
            self.stats['processed'] = len(raw_data)
            
            if not raw_data:
                logger.warning("No data scraped")
                self.stats['end_time'] = datetime.utcnow()
                self._log_ingestion("scrape", "SUCCESS")
                return True
            
            # Process data
            logger.info("🔧 Processing data...")
            processed_data = self.process_data(raw_data)
            
            # Save data
            logger.info("💾 Saving data to database...")
            success = self.save_data(processed_data)
            
            self.stats['end_time'] = datetime.utcnow()
            
            if success:
                self.stats['successful'] = len(processed_data)
                logger.info(f"✅ {self.source_name} scraper completed successfully!")
                logger.info(f"📊 Processed: {self.stats['processed']}, Saved: {self.stats['successful']}")
                self._log_ingestion("scrape", "SUCCESS")
                return True
            else:
                self.stats['failed'] = len(processed_data)
                logger.error(f"❌ {self.source_name} scraper failed to save data")
                self._log_ingestion("scrape", "FAILED", "Failed to save data")
                return False
                
        except Exception as e:
            self.stats['end_time'] = datetime.utcnow()
            self.stats['failed'] = self.stats['processed']
            logger.error(f"❌ {self.source_name} scraper failed: {e}")
            self._log_ingestion("scrape", "FAILED", str(e))
            return False


class BatchProcessor:
    """Utility class for processing data in batches."""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
    
    def process_batches(self, data: List[Dict[str, Any]], processor_func) -> List[Dict[str, Any]]:
        """Process data in batches."""
        results = []
        total_batches = (len(data) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
            
            try:
                batch_results = processor_func(batch)
                results.extend(batch_results)
                logger.info(f"✅ Batch {batch_num} processed successfully")
            except Exception as e:
                logger.error(f"❌ Batch {batch_num} failed: {e}")
                # Continue with next batch
        
        return results
