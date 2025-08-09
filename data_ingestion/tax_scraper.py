"""
Tax Records Scraper for property assessment and tax data.
Collects data from county assessor websites and tax databases.
"""

import re
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from bs4 import BeautifulSoup
from loguru import logger

from data_ingestion.base_scraper import BaseScraper
from database.connection import db_manager
from database.models import Property, TaxRecord


class TaxRecordsScraper(BaseScraper):
    """Tax records scraper for property assessment data."""
    
    def __init__(self):
        super().__init__("TaxRecords")
        self.county_apis = {
            'Harris': {
                'base_url': 'https://www.hcad.org',
                'search_endpoint': '/property-search/',
                'api_key_required': False
            },
            'Dallas': {
                'base_url': 'https://www.dallascad.org',
                'search_endpoint': '/property-search/',
                'api_key_required': False
            },
            'Travis': {
                'base_url': 'https://www.traviscad.org',
                'search_endpoint': '/property-search/',
                'api_key_required': False
            }
        }
    
    def get_required_fields(self) -> List[str]:
        """Return list of required fields for tax data."""
        return ["property_id", "assessed_value", "annual_taxes"]
    
    def scrape_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Scrape tax records data for properties in database."""
        tax_records = []
        
        # Get properties that need tax data
        with db_manager.get_session() as session:
            properties = session.query(Property).filter(
                Property.is_active == True
            ).all()
        
        logger.info(f"Found {len(properties)} properties to scrape tax data for")
        
        for property_obj in properties:
            try:
                # Determine county based on property location
                county = self._get_county_from_location(property_obj.state, property_obj.city)
                
                if county and county in self.county_apis:
                    tax_data = self._scrape_county_tax_data(property_obj, county)
                    if tax_data:
                        tax_data['property_id'] = property_obj.id
                        tax_records.append(tax_data)
                        logger.info(f"✅ Scraped tax data for property {property_obj.id}")
                    else:
                        logger.warning(f"⚠️ No tax data found for property {property_obj.id}")
                else:
                    logger.warning(f"⚠️ No county API configured for {property_obj.city}, {property_obj.state}")
                    
            except Exception as e:
                logger.error(f"❌ Error scraping tax data for property {property_obj.id}: {e}")
                continue
        
        return tax_records
    
    def _get_county_from_location(self, state: str, city: str) -> Optional[str]:
        """Map city/state to county for API selection."""
        county_mapping = {
            'TX': {
                'Houston': 'Harris',
                'Dallas': 'Dallas',
                'Austin': 'Travis',
                'San Antonio': 'Bexar',
                'Fort Worth': 'Tarrant'
            },
            'AZ': {
                'Phoenix': 'Maricopa',
                'Tucson': 'Pima'
            },
            'CA': {
                'Los Angeles': 'Los Angeles',
                'San Francisco': 'San Francisco',
                'San Diego': 'San Diego'
            }
        }
        
        return county_mapping.get(state, {}).get(city)
    
    def _scrape_county_tax_data(self, property_obj: Property, county: str) -> Optional[Dict[str, Any]]:
        """Scrape tax data from specific county assessor website."""
        county_config = self.county_apis[county]
        
        try:
            # Build search URL
            search_url = f"{county_config['base_url']}{county_config['search_endpoint']}"
            
            # Create search parameters
            search_params = {
                'address': property_obj.address,
                'city': property_obj.city,
                'state': property_obj.state,
                'zip': property_obj.zip_code
            }
            
            # Make request to county assessor
            response = self._make_request(search_url, params=search_params)
            
            if response and response.status_code == 200:
                return self._parse_county_response(response.content, county)
            else:
                logger.warning(f"Failed to get response from {county} assessor")
                return None
                
        except Exception as e:
            logger.error(f"Error scraping {county} tax data: {e}")
            return None
    
    def _parse_county_response(self, content: bytes, county: str) -> Optional[Dict[str, Any]]:
        """Parse county assessor response to extract tax data."""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract assessment data
            assessed_value = self._extract_assessed_value(soup, county)
            land_value = self._extract_land_value(soup, county)
            improvement_value = self._extract_improvement_value(soup, county)
            annual_taxes = self._extract_annual_taxes(soup, county)
            tax_rate = self._extract_tax_rate(soup, county)
            assessment_year = self._extract_assessment_year(soup, county)
            
            if assessed_value or annual_taxes:
                return {
                    'assessed_value': assessed_value,
                    'land_value': land_value,
                    'improvement_value': improvement_value,
                    'annual_taxes': annual_taxes,
                    'tax_rate': tax_rate,
                    'assessment_year': assessment_year,
                    'source': f'{county} County Assessor'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing {county} response: {e}")
            return None
    
    def _extract_assessed_value(self, soup: BeautifulSoup, county: str) -> Optional[float]:
        """Extract assessed value from county response."""
        selectors = {
            'Harris': ['#assessed-value', '.assessment-value', '[data-field="assessed_value"]'],
            'Dallas': ['#assessed-value', '.assessment-value', '[data-field="assessed_value"]'],
            'Travis': ['#assessed-value', '.assessment-value', '[data-field="assessed_value"]']
        }
        
        for selector in selectors.get(county, []):
            element = soup.select_one(selector)
            if element:
                value_text = element.get_text(strip=True)
                return self._extract_numeric_value(value_text)
        
        return None
    
    def _extract_land_value(self, soup: BeautifulSoup, county: str) -> Optional[float]:
        """Extract land value from county response."""
        selectors = {
            'Harris': ['#land-value', '.land-value', '[data-field="land_value"]'],
            'Dallas': ['#land-value', '.land-value', '[data-field="land_value"]'],
            'Travis': ['#land-value', '.land-value', '[data-field="land_value"]']
        }
        
        for selector in selectors.get(county, []):
            element = soup.select_one(selector)
            if element:
                value_text = element.get_text(strip=True)
                return self._extract_numeric_value(value_text)
        
        return None
    
    def _extract_improvement_value(self, soup: BeautifulSoup, county: str) -> Optional[float]:
        """Extract improvement value from county response."""
        selectors = {
            'Harris': ['#improvement-value', '.improvement-value', '[data-field="improvement_value"]'],
            'Dallas': ['#improvement-value', '.improvement-value', '[data-field="improvement_value"]'],
            'Travis': ['#improvement-value', '.improvement-value', '[data-field="improvement_value"]']
        }
        
        for selector in selectors.get(county, []):
            element = soup.select_one(selector)
            if element:
                value_text = element.get_text(strip=True)
                return self._extract_numeric_value(value_text)
        
        return None
    
    def _extract_annual_taxes(self, soup: BeautifulSoup, county: str) -> Optional[float]:
        """Extract annual taxes from county response."""
        selectors = {
            'Harris': ['#annual-taxes', '.annual-taxes', '[data-field="annual_taxes"]'],
            'Dallas': ['#annual-taxes', '.annual-taxes', '[data-field="annual_taxes"]'],
            'Travis': ['#annual-taxes', '.annual-taxes', '[data-field="annual_taxes"]']
        }
        
        for selector in selectors.get(county, []):
            element = soup.select_one(selector)
            if element:
                value_text = element.get_text(strip=True)
                return self._extract_numeric_value(value_text)
        
        return None
    
    def _extract_tax_rate(self, soup: BeautifulSoup, county: str) -> Optional[float]:
        """Extract tax rate from county response."""
        selectors = {
            'Harris': ['#tax-rate', '.tax-rate', '[data-field="tax_rate"]'],
            'Dallas': ['#tax-rate', '.tax-rate', '[data-field="tax_rate"]'],
            'Travis': ['#tax-rate', '.tax-rate', '[data-field="tax_rate"]']
        }
        
        for selector in selectors.get(county, []):
            element = soup.select_one(selector)
            if element:
                value_text = element.get_text(strip=True)
                return self._extract_percentage_value(value_text)
        
        return None
    
    def _extract_assessment_year(self, soup: BeautifulSoup, county: str) -> Optional[int]:
        """Extract assessment year from county response."""
        selectors = {
            'Harris': ['#assessment-year', '.assessment-year', '[data-field="assessment_year"]'],
            'Dallas': ['#assessment-year', '.assessment-year', '[data-field="assessment_year"]'],
            'Travis': ['#assessment-year', '.assessment-year', '[data-field="assessment_year"]']
        }
        
        for selector in selectors.get(county, []):
            element = soup.select_one(selector)
            if element:
                value_text = element.get_text(strip=True)
                try:
                    return int(value_text.strip())
                except ValueError:
                    continue
        
        return datetime.now().year
    
    def _extract_numeric_value(self, text: str) -> Optional[float]:
        """Extract numeric value from text, removing currency symbols and commas."""
        if not text:
            return None
        
        # Remove currency symbols, commas, and other non-numeric characters
        clean_text = re.sub(r'[^\d.]', '', text)
        try:
            return float(clean_text) if clean_text else None
        except ValueError:
            return None
    
    def _extract_percentage_value(self, text: str) -> Optional[float]:
        """Extract percentage value from text."""
        if not text:
            return None
        
        # Remove % symbol and convert to decimal
        clean_text = re.sub(r'[^\d.]', '', text)
        try:
            value = float(clean_text) if clean_text else None
            return value / 100 if value else None  # Convert percentage to decimal
        except ValueError:
            return None
    
    def process_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean raw tax data."""
        processed_data = []
        
        for tax_data in raw_data:
            try:
                # Validate required fields
                if not self._validate_data(tax_data):
                    continue
                
                # Clean and standardize data
                cleaned_data = self._clean_tax_data(tax_data)
                processed_data.append(cleaned_data)
                
            except Exception as e:
                logger.error(f"Error processing tax data: {e}")
                continue
        
        return processed_data
    
    def _clean_tax_data(self, tax_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize tax data."""
        cleaned_data = tax_data.copy()
        
        # Ensure numeric fields are properly formatted
        numeric_fields = ['assessed_value', 'land_value', 'improvement_value', 'annual_taxes', 'tax_rate']
        for field in numeric_fields:
            if field in cleaned_data and cleaned_data[field] is not None:
                try:
                    cleaned_data[field] = float(cleaned_data[field])
                except (ValueError, TypeError):
                    cleaned_data[field] = None
        
        # Ensure assessment year is valid
        if 'assessment_year' in cleaned_data and cleaned_data['assessment_year']:
            try:
                year = int(cleaned_data['assessment_year'])
                if year < 1900 or year > datetime.now().year + 1:
                    cleaned_data['assessment_year'] = datetime.now().year
            except (ValueError, TypeError):
                cleaned_data['assessment_year'] = datetime.now().year
        
        return cleaned_data
    
    def save_data(self, processed_data: List[Dict[str, Any]]) -> bool:
        """Save processed tax data to database."""
        try:
            with db_manager.get_session() as session:
                for tax_data in processed_data:
                    # Check if tax record already exists
                    existing_record = session.query(TaxRecord).filter(
                        TaxRecord.property_id == tax_data['property_id']
                    ).first()
                    
                    if existing_record:
                        # Update existing record
                        for key, value in tax_data.items():
                            if key != 'property_id':  # Don't update the foreign key
                                setattr(existing_record, key, value)
                        existing_record.updated_at = datetime.utcnow()
                    else:
                        # Create new record
                        tax_record = TaxRecord(**tax_data)
                        session.add(tax_record)
                
                session.commit()
                logger.info(f"✅ Saved {len(processed_data)} tax records")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error saving tax data: {e}")
            return False


class MockTaxRecordsScraper(TaxRecordsScraper):
    """Mock tax records scraper for testing and development."""
    
    def scrape_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Generate mock tax data for testing."""
        logger.info("🔧 Using mock tax data for development")
        
        # Get properties from database
        with db_manager.get_session() as session:
            # Query only the needed columns to avoid session issues
            properties_data = session.query(
                Property.id, Property.list_price
            ).filter(
                Property.is_active == True
            ).limit(50).all()  # Limit for testing
        
        mock_tax_records = []
        
        for property_id, list_price in properties_data:
            # Generate realistic mock data based on property characteristics
            assessed_value = list_price * 0.85 if list_price else None
            land_value = assessed_value * 0.3 if assessed_value else None
            improvement_value = assessed_value * 0.7 if assessed_value else None
            annual_taxes = assessed_value * 0.025 if assessed_value else None  # 2.5% tax rate
            tax_rate = 0.025  # 2.5%
            
            mock_tax_records.append({
                'property_id': property_id,
                'assessed_value': assessed_value,
                'land_value': land_value,
                'improvement_value': improvement_value,
                'annual_taxes': annual_taxes,
                'tax_rate': tax_rate,
                'assessment_year': datetime.now().year,
                'source': 'Mock Tax Data'
            })
        
        return mock_tax_records
