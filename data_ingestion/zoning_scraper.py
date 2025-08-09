"""
Zoning Data Scraper for land use and zoning information.
Collects data from municipal planning departments and zoning databases.
"""

import re
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from bs4 import BeautifulSoup
from loguru import logger

from data_ingestion.base_scraper import BaseScraper
from database.connection import db_manager
from database.models import Property, ZoningData


class ZoningDataScraper(BaseScraper):
    """Zoning data scraper for land use and development information."""
    
    def __init__(self):
        super().__init__("ZoningData")
        self.city_apis = {
            'Houston': {
                'base_url': 'https://www.houstontx.gov',
                'zoning_endpoint': '/planning/zoning/',
                'api_key_required': False
            },
            'Dallas': {
                'base_url': 'https://dallascityhall.com',
                'zoning_endpoint': '/departments/planning-urban-design',
                'api_key_required': False
            },
            'Austin': {
                'base_url': 'https://www.austintexas.gov',
                'zoning_endpoint': '/department/planning-zoning',
                'api_key_required': False
            },
            'Phoenix': {
                'base_url': 'https://www.phoenix.gov',
                'zoning_endpoint': '/planning/zoning',
                'api_key_required': False
            }
        }
        
        # Common zoning codes and descriptions
        self.zoning_codes = {
            'SF-1': 'Single Family Residential - Large Lot',
            'SF-2': 'Single Family Residential - Standard Lot',
            'SF-3': 'Single Family Residential - Small Lot',
            'MF-1': 'Multi-Family Residential - Low Density',
            'MF-2': 'Multi-Family Residential - Medium Density',
            'MF-3': 'Multi-Family Residential - High Density',
            'C-1': 'Commercial - Neighborhood',
            'C-2': 'Commercial - General',
            'C-3': 'Commercial - Heavy',
            'I-1': 'Industrial - Light',
            'I-2': 'Industrial - Heavy',
            'AG': 'Agricultural',
            'P': 'Public/Institutional'
        }
    
    def get_required_fields(self) -> List[str]:
        """Return list of required fields for zoning data."""
        return ["property_id", "zoning_code"]
    
    def scrape_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Scrape zoning data for properties in database."""
        zoning_records = []
        
        # Get properties that need zoning data
        with db_manager.get_session() as session:
            properties = session.query(Property).filter(
                Property.is_active == True
            ).all()
        
        logger.info(f"Found {len(properties)} properties to scrape zoning data for")
        
        for property_obj in properties:
            try:
                # Check if city has zoning API
                if property_obj.city in self.city_apis:
                    zoning_data = self._scrape_city_zoning_data(property_obj)
                    if zoning_data:
                        zoning_data['property_id'] = property_obj.id
                        zoning_records.append(zoning_data)
                        logger.info(f"✅ Scraped zoning data for property {property_obj.id}")
                    else:
                        logger.warning(f"⚠️ No zoning data found for property {property_obj.id}")
                else:
                    logger.warning(f"⚠️ No zoning API configured for {property_obj.city}")
                    
            except Exception as e:
                logger.error(f"❌ Error scraping zoning data for property {property_obj.id}: {e}")
                continue
        
        return zoning_records
    
    def _scrape_city_zoning_data(self, property_obj: Property) -> Optional[Dict[str, Any]]:
        """Scrape zoning data from specific city planning department."""
        city_config = self.city_apis[property_obj.city]
        
        try:
            # Build search URL
            search_url = f"{city_config['base_url']}{city_config['zoning_endpoint']}"
            
            # Create search parameters
            search_params = {
                'address': property_obj.address,
                'city': property_obj.city,
                'state': property_obj.state,
                'zip': property_obj.zip_code
            }
            
            # Make request to city planning department
            response = self._make_request(search_url, params=search_params)
            
            if response and response.status_code == 200:
                return self._parse_city_response(response.content, property_obj.city)
            else:
                logger.warning(f"Failed to get response from {property_obj.city} planning department")
                return None
                
        except Exception as e:
            logger.error(f"Error scraping {property_obj.city} zoning data: {e}")
            return None
    
    def _parse_city_response(self, content: bytes, city: str) -> Optional[Dict[str, Any]]:
        """Parse city planning department response to extract zoning data."""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract zoning information
            zoning_code = self._extract_zoning_code(soup, city)
            zoning_description = self._extract_zoning_description(soup, city)
            land_use = self._extract_land_use(soup, city)
            max_density = self._extract_max_density(soup, city)
            max_height = self._extract_max_height(soup, city)
            setback_requirements = self._extract_setback_requirements(soup, city)
            permitted_uses = self._extract_permitted_uses(soup, city)
            conditional_uses = self._extract_conditional_uses(soup, city)
            
            if zoning_code:
                return {
                    'zoning_code': zoning_code,
                    'zoning_description': zoning_description,
                    'land_use': land_use,
                    'max_density': max_density,
                    'max_height': max_height,
                    'setback_requirements': setback_requirements,
                    'permitted_uses': permitted_uses,
                    'conditional_uses': conditional_uses,
                    'source': f'{city} Planning Department'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing {city} response: {e}")
            return None
    
    def _extract_zoning_code(self, soup: BeautifulSoup, city: str) -> Optional[str]:
        """Extract zoning code from city response."""
        selectors = {
            'Houston': ['#zoning-code', '.zoning-code', '[data-field="zoning_code"]'],
            'Dallas': ['#zoning-code', '.zoning-code', '[data-field="zoning_code"]'],
            'Austin': ['#zoning-code', '.zoning-code', '[data-field="zoning_code"]'],
            'Phoenix': ['#zoning-code', '.zoning-code', '[data-field="zoning_code"]']
        }
        
        for selector in selectors.get(city, []):
            element = soup.select_one(selector)
            if element:
                zoning_code = element.get_text(strip=True)
                if zoning_code and zoning_code in self.zoning_codes:
                    return zoning_code
        
        return None
    
    def _extract_zoning_description(self, soup: BeautifulSoup, city: str) -> Optional[str]:
        """Extract zoning description from city response."""
        selectors = {
            'Houston': ['#zoning-description', '.zoning-description', '[data-field="zoning_description"]'],
            'Dallas': ['#zoning-description', '.zoning-description', '[data-field="zoning_description"]'],
            'Austin': ['#zoning-description', '.zoning-description', '[data-field="zoning_description"]'],
            'Phoenix': ['#zoning-description', '.zoning-description', '[data-field="zoning_description"]']
        }
        
        for selector in selectors.get(city, []):
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        return None
    
    def _extract_land_use(self, soup: BeautifulSoup, city: str) -> Optional[str]:
        """Extract land use from city response."""
        selectors = {
            'Houston': ['#land-use', '.land-use', '[data-field="land_use"]'],
            'Dallas': ['#land-use', '.land-use', '[data-field="land_use"]'],
            'Austin': ['#land-use', '.land-use', '[data-field="land_use"]'],
            'Phoenix': ['#land-use', '.land-use', '[data-field="land_use"]']
        }
        
        for selector in selectors.get(city, []):
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        return None
    
    def _extract_max_density(self, soup: BeautifulSoup, city: str) -> Optional[float]:
        """Extract maximum density from city response."""
        selectors = {
            'Houston': ['#max-density', '.max-density', '[data-field="max_density"]'],
            'Dallas': ['#max-density', '.max-density', '[data-field="max_density"]'],
            'Austin': ['#max-density', '.max-density', '[data-field="max_density"]'],
            'Phoenix': ['#max-density', '.max-density', '[data-field="max_density"]']
        }
        
        for selector in selectors.get(city, []):
            element = soup.select_one(selector)
            if element:
                value_text = element.get_text(strip=True)
                return self._extract_numeric_value(value_text)
        
        return None
    
    def _extract_max_height(self, soup: BeautifulSoup, city: str) -> Optional[float]:
        """Extract maximum height from city response."""
        selectors = {
            'Houston': ['#max-height', '.max-height', '[data-field="max_height"]'],
            'Dallas': ['#max-height', '.max-height', '[data-field="max_height"]'],
            'Austin': ['#max-height', '.max-height', '[data-field="max_height"]'],
            'Phoenix': ['#max-height', '.max-height', '[data-field="max_height"]']
        }
        
        for selector in selectors.get(city, []):
            element = soup.select_one(selector)
            if element:
                value_text = element.get_text(strip=True)
                return self._extract_numeric_value(value_text)
        
        return None
    
    def _extract_setback_requirements(self, soup: BeautifulSoup, city: str) -> Optional[Dict[str, float]]:
        """Extract setback requirements from city response."""
        selectors = {
            'Houston': ['#setback-requirements', '.setback-requirements', '[data-field="setback_requirements"]'],
            'Dallas': ['#setback-requirements', '.setback-requirements', '[data-field="setback_requirements"]'],
            'Austin': ['#setback-requirements', '.setback-requirements', '[data-field="setback_requirements"]'],
            'Phoenix': ['#setback-requirements', '.setback-requirements', '[data-field="setback_requirements"]']
        }
        
        for selector in selectors.get(city, []):
            element = soup.select_one(selector)
            if element:
                # Try to parse JSON or structured data
                try:
                    data = json.loads(element.get_text(strip=True))
                    return data
                except json.JSONDecodeError:
                    # Try to parse from text
                    text = element.get_text(strip=True)
                    return self._parse_setback_text(text)
        
        return None
    
    def _extract_permitted_uses(self, soup: BeautifulSoup, city: str) -> Optional[List[str]]:
        """Extract permitted uses from city response."""
        selectors = {
            'Houston': ['#permitted-uses', '.permitted-uses', '[data-field="permitted_uses"]'],
            'Dallas': ['#permitted-uses', '.permitted-uses', '[data-field="permitted_uses"]'],
            'Austin': ['#permitted-uses', '.permitted-uses', '[data-field="permitted_uses"]'],
            'Phoenix': ['#permitted-uses', '.permitted-uses', '[data-field="permitted_uses"]']
        }
        
        for selector in selectors.get(city, []):
            element = soup.select_one(selector)
            if element:
                # Try to parse JSON or structured data
                try:
                    data = json.loads(element.get_text(strip=True))
                    return data if isinstance(data, list) else None
                except json.JSONDecodeError:
                    # Try to parse from text
                    text = element.get_text(strip=True)
                    return self._parse_uses_text(text)
        
        return None
    
    def _extract_conditional_uses(self, soup: BeautifulSoup, city: str) -> Optional[List[str]]:
        """Extract conditional uses from city response."""
        selectors = {
            'Houston': ['#conditional-uses', '.conditional-uses', '[data-field="conditional_uses"]'],
            'Dallas': ['#conditional-uses', '.conditional-uses', '[data-field="conditional_uses"]'],
            'Austin': ['#conditional-uses', '.conditional-uses', '[data-field="conditional_uses"]'],
            'Phoenix': ['#conditional-uses', '.conditional-uses', '[data-field="conditional_uses"]']
        }
        
        for selector in selectors.get(city, []):
            element = soup.select_one(selector)
            if element:
                # Try to parse JSON or structured data
                try:
                    data = json.loads(element.get_text(strip=True))
                    return data if isinstance(data, list) else None
                except json.JSONDecodeError:
                    # Try to parse from text
                    text = element.get_text(strip=True)
                    return self._parse_uses_text(text)
        
        return None
    
    def _extract_numeric_value(self, text: str) -> Optional[float]:
        """Extract numeric value from text."""
        if not text:
            return None
        
        # Remove non-numeric characters except decimal point
        clean_text = re.sub(r'[^\d.]', '', text)
        try:
            return float(clean_text) if clean_text else None
        except ValueError:
            return None
    
    def _parse_setback_text(self, text: str) -> Optional[Dict[str, float]]:
        """Parse setback requirements from text."""
        if not text:
            return None
        
        setbacks = {}
        
        # Look for common setback patterns
        patterns = {
            'front': r'front[:\s]*(\d+(?:\.\d+)?)',
            'side': r'side[:\s]*(\d+(?:\.\d+)?)',
            'rear': r'rear[:\s]*(\d+(?:\.\d+)?)'
        }
        
        for setback_type, pattern in patterns.items():
            match = re.search(pattern, text.lower())
            if match:
                try:
                    setbacks[setback_type] = float(match.group(1))
                except ValueError:
                    continue
        
        return setbacks if setbacks else None
    
    def _parse_uses_text(self, text: str) -> Optional[List[str]]:
        """Parse permitted/conditional uses from text."""
        if not text:
            return None
        
        # Split by common delimiters
        uses = []
        for delimiter in [',', ';', '\n', '•', '-']:
            if delimiter in text:
                uses = [use.strip() for use in text.split(delimiter) if use.strip()]
                break
        
        if not uses:
            uses = [text.strip()]
        
        return uses if uses else None
    
    def process_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean raw zoning data."""
        processed_data = []
        
        for zoning_data in raw_data:
            try:
                # Validate required fields
                if not self._validate_data(zoning_data):
                    continue
                
                # Clean and standardize data
                cleaned_data = self._clean_zoning_data(zoning_data)
                processed_data.append(cleaned_data)
                
            except Exception as e:
                logger.error(f"Error processing zoning data: {e}")
                continue
        
        return processed_data
    
    def _clean_zoning_data(self, zoning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize zoning data."""
        cleaned_data = zoning_data.copy()
        
        # Ensure numeric fields are properly formatted
        numeric_fields = ['max_density', 'max_height']
        for field in numeric_fields:
            if field in cleaned_data and cleaned_data[field] is not None:
                try:
                    cleaned_data[field] = float(cleaned_data[field])
                except (ValueError, TypeError):
                    cleaned_data[field] = None
        
        # Ensure JSON fields are properly formatted
        json_fields = ['setback_requirements', 'permitted_uses', 'conditional_uses']
        for field in json_fields:
            if field in cleaned_data and cleaned_data[field] is not None:
                if isinstance(cleaned_data[field], str):
                    try:
                        cleaned_data[field] = json.loads(cleaned_data[field])
                    except json.JSONDecodeError:
                        # If it's not valid JSON, keep as string
                        pass
        
        # Add zoning description if not provided
        if 'zoning_code' in cleaned_data and cleaned_data['zoning_code']:
            if 'zoning_description' not in cleaned_data or not cleaned_data['zoning_description']:
                cleaned_data['zoning_description'] = self.zoning_codes.get(
                    cleaned_data['zoning_code'], 
                    'Unknown Zoning Code'
                )
        
        return cleaned_data
    
    def save_data(self, processed_data: List[Dict[str, Any]]) -> bool:
        """Save processed zoning data to database."""
        try:
            with db_manager.get_session() as session:
                for zoning_data in processed_data:
                    # Check if zoning record already exists
                    existing_record = session.query(ZoningData).filter(
                        ZoningData.property_id == zoning_data['property_id']
                    ).first()
                    
                    if existing_record:
                        # Update existing record
                        for key, value in zoning_data.items():
                            if key != 'property_id':  # Don't update the foreign key
                                if isinstance(value, (dict, list)):
                                    value = json.dumps(value)
                                setattr(existing_record, key, value)
                        existing_record.updated_at = datetime.utcnow()
                    else:
                        # Create new record
                        # Convert dict/list fields to JSON strings for storage
                        for key, value in zoning_data.items():
                            if isinstance(value, (dict, list)):
                                zoning_data[key] = json.dumps(value)
                        
                        zoning_record = ZoningData(**zoning_data)
                        session.add(zoning_record)
                
                session.commit()
                logger.info(f"✅ Saved {len(processed_data)} zoning records")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error saving zoning data: {e}")
            return False


class MockZoningDataScraper(ZoningDataScraper):
    """Mock zoning data scraper for testing and development."""
    
    def scrape_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Generate mock zoning data for testing."""
        logger.info("🔧 Using mock zoning data for development")
        
        # Get properties from database
        with db_manager.get_session() as session:
            # Query only the needed columns to avoid session issues
            properties_data = session.query(
                Property.id, Property.property_type
            ).filter(
                Property.is_active == True
            ).limit(50).all()  # Limit for testing
        
        mock_zoning_records = []
        
        for property_id, property_type in properties_data:
            # Generate realistic mock data based on property characteristics
            if property_type and 'Single Family' in property_type:
                zoning_code = 'SF-2'
                land_use = 'Single Family Residential'
                max_density = 4.0  # units per acre
                max_height = 35.0  # feet
                permitted_uses = ['Single Family Residential', 'Accessory Structures']
                conditional_uses = ['Home Office', 'Day Care']
            elif property_type and 'Multi' in property_type:
                zoning_code = 'MF-2'
                land_use = 'Multi-Family Residential'
                max_density = 25.0  # units per acre
                max_height = 45.0  # feet
                permitted_uses = ['Multi-Family Residential', 'Accessory Structures']
                conditional_uses = ['Community Center', 'Retail']
            else:
                zoning_code = 'C-1'
                land_use = 'Commercial'
                max_density = 15.0  # units per acre
                max_height = 40.0  # feet
                permitted_uses = ['Commercial', 'Office']
                conditional_uses = ['Restaurant', 'Retail']
            
            setback_requirements = {
                'front': 20.0,
                'side': 8.0,
                'rear': 25.0
            }
            
            mock_zoning_records.append({
                'property_id': property_id,
                'zoning_code': zoning_code,
                'zoning_description': self.zoning_codes.get(zoning_code, 'Unknown'),
                'land_use': land_use,
                'max_density': max_density,
                'max_height': max_height,
                'setback_requirements': setback_requirements,
                'permitted_uses': permitted_uses,
                'conditional_uses': conditional_uses,
                'source': 'Mock Zoning Data'
            })
        
        return mock_zoning_records
