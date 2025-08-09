"""
MLS (Multiple Listing Service) scraper for property listings.
Supports multiple MLS sources and scraping methods.
"""

import re
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from bs4 import BeautifulSoup
from loguru import logger

from data_ingestion.base_scraper import BaseScraper
from database.connection import db_manager
from database.models import Property


class MLSScraper(BaseScraper):
    """MLS scraper for property listings."""
    
    def __init__(self):
        super().__init__("MLS")
        self.property_cache = set()  # Cache to avoid duplicates
    
    def get_required_fields(self) -> List[str]:
        """Return list of required fields for MLS data."""
        return ["address", "city", "state", "zip_code"]
    
    def scrape_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Scrape MLS data from configured sources."""
        all_properties = []
        
        # Scrape from multiple sources
        sources = [
            self._scrape_zillow,
            self._scrape_redfin,
            self._scrape_realtor,
        ]
        
        for source_func in sources:
            try:
                logger.info(f"Scraping from {source_func.__name__}...")
                properties = source_func(**kwargs)
                all_properties.extend(properties)
                logger.info(f"✅ Scraped {len(properties)} properties from {source_func.__name__}")
            except Exception as e:
                logger.error(f"❌ Failed to scrape from {source_func.__name__}: {e}")
        
        return all_properties
    
    def _scrape_zillow(self, **kwargs) -> List[Dict[str, Any]]:
        """Scrape property data from Zillow."""
        properties = []
        
        # Example Zillow scraping logic
        # In a real implementation, you would need to handle authentication and rate limiting
        search_urls = [
            "https://www.zillow.com/homes/for_sale/",
            "https://www.zillow.com/homes/for_rent/",
        ]
        
        for url in search_urls:
            try:
                response = self._make_request(url)
                if response:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    # Extract property listings from HTML
                    # This is a simplified example - real implementation would be more complex
                    property_elements = soup.find_all('div', class_='property-card')
                    
                    for element in property_elements:
                        property_data = self._extract_zillow_property(element)
                        if property_data:
                            properties.append(property_data)
                            
            except Exception as e:
                logger.error(f"Error scraping Zillow: {e}")
        
        return properties
    
    def _extract_zillow_property(self, element) -> Optional[Dict[str, Any]]:
        """Extract property data from Zillow HTML element."""
        try:
            # Extract address
            address_elem = element.find('address')
            if not address_elem:
                return None
            
            address_text = address_elem.get_text(strip=True)
            address_parts = self._parse_address(address_text)
            
            # Extract price
            price_elem = element.find('span', class_='price')
            price = self._extract_price(price_elem.get_text() if price_elem else "")
            
            # Extract property details
            details = self._extract_property_details(element)
            
            property_data = {
                'address': address_parts.get('street'),
                'city': address_parts.get('city'),
                'state': address_parts.get('state'),
                'zip_code': address_parts.get('zip'),
                'list_price': price,
                'bedrooms': details.get('bedrooms'),
                'bathrooms': details.get('bathrooms'),
                'square_feet': details.get('square_feet'),
                'property_type': details.get('property_type'),
                'source': 'Zillow',
                'mls_id': self._generate_mls_id(address_parts),
            }
            
            return property_data
            
        except Exception as e:
            logger.error(f"Error extracting Zillow property: {e}")
            return None
    
    def _scrape_redfin(self, **kwargs) -> List[Dict[str, Any]]:
        """Scrape property data from Redfin."""
        properties = []
        
        # Redfin scraping logic would go here
        # Similar to Zillow but with Redfin-specific selectors
        logger.info("Redfin scraping not implemented yet")
        
        return properties
    
    def _scrape_realtor(self, **kwargs) -> List[Dict[str, Any]]:
        """Scrape property data from Realtor.com."""
        properties = []
        
        # Realtor.com scraping logic would go here
        logger.info("Realtor.com scraping not implemented yet")
        
        return properties
    
    def _parse_address(self, address_text: str) -> Dict[str, str]:
        """Parse address string into components."""
        # Simple address parsing - in production, use a proper address parser
        parts = address_text.split(',')
        if len(parts) >= 3:
            street = parts[0].strip()
            city = parts[1].strip()
            state_zip = parts[2].strip()
            
            # Extract state and zip
            state_zip_parts = state_zip.split()
            state = state_zip_parts[0] if state_zip_parts else ""
            zip_code = state_zip_parts[1] if len(state_zip_parts) > 1 else ""
            
            return {
                'street': street,
                'city': city,
                'state': state,
                'zip': zip_code
            }
        
        return {}
    
    def _extract_price(self, price_text: str) -> Optional[float]:
        """Extract numeric price from price text."""
        if not price_text:
            return None
        
        # Remove non-numeric characters except decimal point
        price_clean = re.sub(r'[^\d.]', '', price_text)
        try:
            return float(price_clean) if price_clean else None
        except ValueError:
            return None
    
    def _extract_property_details(self, element) -> Dict[str, Any]:
        """Extract property details from HTML element."""
        details = {}
        
        # Extract bedrooms
        bed_elem = element.find('span', class_='bedrooms')
        if bed_elem:
            bed_text = bed_elem.get_text()
            details['bedrooms'] = self._extract_number(bed_text)
        
        # Extract bathrooms
        bath_elem = element.find('span', class_='bathrooms')
        if bath_elem:
            bath_text = bath_elem.get_text()
            details['bathrooms'] = self._extract_number(bath_text)
        
        # Extract square footage
        sqft_elem = element.find('span', class_='square-feet')
        if sqft_elem:
            sqft_text = sqft_elem.get_text()
            details['square_feet'] = self._extract_number(sqft_text)
        
        # Extract property type
        type_elem = element.find('span', class_='property-type')
        if type_elem:
            details['property_type'] = type_elem.get_text(strip=True)
        
        return details
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numeric value from text."""
        if not text:
            return None
        
        numbers = re.findall(r'\d+\.?\d*', text)
        try:
            return float(numbers[0]) if numbers else None
        except (ValueError, IndexError):
            return None
    
    def _generate_mls_id(self, address_parts: Dict[str, str]) -> str:
        """Generate a unique MLS ID from address."""
        street = address_parts.get('street', '').replace(' ', '').upper()
        city = address_parts.get('city', '').replace(' ', '').upper()
        state = address_parts.get('state', '').upper()
        zip_code = address_parts.get('zip', '')
        
        return f"{street}_{city}_{state}_{zip_code}"
    
    def process_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean raw MLS data."""
        processed_data = []
        
        for property_data in raw_data:
            try:
                # Validate required fields
                if not self._validate_data(property_data):
                    continue
                
                # Clean and standardize data
                cleaned_data = self._clean_property_data(property_data)
                
                # Check for duplicates
                if self._is_duplicate(cleaned_data):
                    continue
                
                processed_data.append(cleaned_data)
                
            except Exception as e:
                logger.error(f"Error processing property data: {e}")
                continue
        
        return processed_data
    
    def _clean_property_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize property data."""
        cleaned = data.copy()
        
        # Clean address
        if 'address' in cleaned:
            cleaned['address'] = cleaned['address'].strip().title()
        
        # Clean city
        if 'city' in cleaned:
            cleaned['city'] = cleaned['city'].strip().title()
        
        # Clean state
        if 'state' in cleaned:
            cleaned['state'] = cleaned['state'].strip().upper()
        
        # Clean zip code
        if 'zip_code' in cleaned:
            cleaned['zip_code'] = cleaned['zip_code'].strip()
        
        # Convert numeric fields
        numeric_fields = ['list_price', 'bedrooms', 'bathrooms', 'square_feet']
        for field in numeric_fields:
            if field in cleaned and cleaned[field] is not None:
                try:
                    cleaned[field] = float(cleaned[field])
                except (ValueError, TypeError):
                    cleaned[field] = None
        
        return cleaned
    
    def _is_duplicate(self, property_data: Dict[str, Any]) -> bool:
        """Check if property is a duplicate based on MLS ID or address."""
        mls_id = property_data.get('mls_id')
        address = property_data.get('address')
        
        if mls_id and mls_id in self.property_cache:
            return True
        
        # Check database for existing property
        try:
            with db_manager.get_session() as session:
                existing = session.query(Property).filter(
                    (Property.mls_id == mls_id) | 
                    (Property.address == address)
                ).first()
                
                if existing:
                    return True
                    
        except Exception as e:
            logger.error(f"Error checking for duplicates: {e}")
        
        # Add to cache
        if mls_id:
            self.property_cache.add(mls_id)
        
        return False
    
    def save_data(self, processed_data: List[Dict[str, Any]]) -> bool:
        """Save processed MLS data to database."""
        try:
            with db_manager.get_session() as session:
                for property_data in processed_data:
                    # Create Property object
                    property_obj = Property(**property_data)
                    session.add(property_obj)
                
                session.commit()
                logger.info(f"✅ Saved {len(processed_data)} properties to database")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to save MLS data: {e}")
            return False
