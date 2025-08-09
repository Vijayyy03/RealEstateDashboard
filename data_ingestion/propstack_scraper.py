"""Propstack API scraper for real estate data in India.

This module implements a scraper for the Propstack API, which provides
comprehensive real estate data for the Indian market.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger

from config.settings import settings
from database.connection import db_manager
from database.models import Property, UnderwritingData
from data_ingestion.base_scraper import BaseScraper


class PropstackScraper(BaseScraper):
    """Scraper for Propstack API data."""
    
    def __init__(self):
        super().__init__(source_name="Propstack")
        self.api_key = settings.external_apis.propstack_api_key
        self.base_url = settings.external_apis.propstack_base_url
        
        if not self.api_key:
            logger.warning("Propstack API key not configured. Using demo data.")
    
    def get_required_fields(self) -> List[str]:
        """Return list of required fields for this scraper."""
        return [
            "property_id",
            "property_type",
            "address",
            "city",
            "price"
        ]
    
    def scrape_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Scrape data from Propstack API."""
        logger.info("Fetching data from Propstack API")
        
        # If API key is not configured, use demo data
        if not self.api_key:
            return self._get_demo_data()
        
        # Parameters for API request
        params = {
            "api_key": self.api_key,
            "city": kwargs.get("city", "Mumbai"),
            "property_type": kwargs.get("property_type", "Residential"),
            "limit": kwargs.get("limit", 100)
        }
        
        # Make API request
        endpoint = f"{self.base_url}/properties/search"
        response = self._make_request(endpoint, params=params)
        
        if not response:
            logger.error("Failed to fetch data from Propstack API")
            return []
        
        try:
            data = response.json()
            properties = data.get("properties", [])
            logger.info(f"Retrieved {len(properties)} properties from Propstack API")
            return properties
        except Exception as e:
            logger.error(f"Error parsing Propstack API response: {e}")
            return []
    
    def process_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean raw scraped data."""
        processed_data = []
        
        for item in raw_data:
            try:
                # Skip items missing required fields
                if not self._validate_data(item):
                    continue
                
                # Process and standardize the data
                processed_item = {
                    "source": "Propstack",
                    "source_id": item.get("property_id"),
                    "property_type": item.get("property_type"),
                    "address": item.get("address"),
                    "city": item.get("city"),
                    "state": item.get("state"),
                    "zip_code": item.get("pincode"),
                    "price": float(item.get("price", 0)),
                    "bedrooms": int(item.get("bedrooms", 0)),
                    "bathrooms": int(item.get("bathrooms", 0)),
                    "sqft": float(item.get("carpet_area", 0)),
                    "year_built": item.get("year_built"),
                    "latitude": float(item.get("latitude", 0)) if item.get("latitude") else None,
                    "longitude": float(item.get("longitude", 0)) if item.get("longitude") else None,
                    "is_active": True,
                    
                    # Financial data
                    "monthly_rent": float(item.get("monthly_rent", 0)) if item.get("monthly_rent") else None,
                    "cap_rate": float(item.get("cap_rate", 0)) if item.get("cap_rate") else None,
                    "noi": float(item.get("noi", 0)) if item.get("noi") else None,
                    
                    # Additional fields specific to Indian market (stored as features)
                    "amenities": [
                        f"Furnishing: {item.get('furnishing')}" if item.get('furnishing') else None,
                        f"Facing: {item.get('facing')}" if item.get('facing') else None,
                        f"Floor: {item.get('floor')}" if item.get('floor') else None,
                        f"Total Floors: {item.get('total_floors')}" if item.get('total_floors') else None
                    ] + item.get("amenities", [])
                }
                
                # Remove None values from amenities list
                processed_item["amenities"] = [a for a in processed_item["amenities"] if a is not None]
                
                processed_data.append(processed_item)
                
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                continue
        
        logger.info(f"Processed {len(processed_data)} properties")
        return processed_data
    
    def save_data(self, processed_data: List[Dict[str, Any]]) -> bool:
        """Save processed data to database."""
        from database.models import Property, UnderwritingData
        
        if not processed_data:
            logger.warning("No data to save")
            return True
        
        try:
            with db_manager.get_session() as session:
                saved_count = 0
                updated_count = 0
                
                for item in processed_data:
                    try:
                        # Check if property already exists by source and source_id
                        existing_property = session.query(Property).filter(
                            Property.source == item["source"],
                            Property.mls_id == item["source_id"]
                        ).first()
                        
                        if existing_property:
                            # Update existing property
                            existing_property.property_type = item["property_type"]
                            existing_property.address = item["address"]
                            existing_property.city = item["city"]
                            existing_property.state = item["state"]
                            existing_property.zip_code = item["zip_code"]
                            existing_property.bedrooms = item["bedrooms"]
                            existing_property.bathrooms = item["bathrooms"]
                            existing_property.square_feet = item["sqft"]
                            existing_property.year_built = item["year_built"]
                            existing_property.list_price = item["price"]
                            existing_property.monthly_rent = item.get("monthly_rent")
                            existing_property.annual_rent = item.get("monthly_rent") * 12 if item.get("monthly_rent") else None
                            existing_property.features = json.dumps(item.get("amenities", [])) if item.get("amenities") else None
                            existing_property.latitude = item["latitude"]
                            existing_property.longitude = item["longitude"]
                            existing_property.is_active = item["is_active"]
                            existing_property.updated_at = datetime.utcnow()
                            
                            # Update underwriting data if available
                            underwriting = session.query(UnderwritingData).filter(
                                UnderwritingData.property_id == existing_property.id
                            ).first()
                            
                            if underwriting and any(item.get(field) for field in ["monthly_rent", "cap_rate", "noi"]):
                                underwriting.gross_rental_income = item.get("monthly_rent") * 12 if item.get("monthly_rent") else None
                                underwriting.total_income = item.get("monthly_rent") * 12 if item.get("monthly_rent") else None
                                underwriting.noi = item.get("noi")
                                underwriting.cap_rate = item.get("cap_rate")
                                underwriting.purchase_price = item["price"]
                                underwriting.updated_at = datetime.utcnow()
                            elif any(item.get(field) for field in ["monthly_rent", "cap_rate", "noi"]):
                                # Create new underwriting data if it doesn't exist
                                underwriting_data = UnderwritingData(
                                    property_id=existing_property.id,
                                    gross_rental_income=item.get("monthly_rent") * 12 if item.get("monthly_rent") else None,
                                    other_income=0,  # Default to 0
                                    total_income=item.get("monthly_rent") * 12 if item.get("monthly_rent") else None,
                                    noi=item.get("noi"),
                                    cap_rate=item.get("cap_rate"),
                                    purchase_price=item["price"],
                                    created_at=datetime.utcnow(),
                                    updated_at=datetime.utcnow()
                                )
                                session.add(underwriting_data)
                            
                            updated_count += 1
                            logger.debug(f"Updated property {existing_property.id}")
                        else:
                            # Create new property
                            property_obj = Property(
                                mls_id=item["source_id"],
                                source=item["source"],
                                address=item["address"],
                                city=item["city"],
                                state=item["state"],
                                zip_code=item["zip_code"],
                                county=None,  # Not provided by Propstack
                                property_type=item["property_type"],
                                bedrooms=item["bedrooms"],
                                bathrooms=item["bathrooms"],
                                square_feet=item["sqft"],
                                lot_size=None,  # Not provided by Propstack
                                year_built=item["year_built"],
                                list_price=item["price"],
                                sold_price=None,  # Not provided by Propstack
                                estimated_value=None,  # Not provided by Propstack
                                monthly_rent=item.get("monthly_rent"),
                                annual_rent=item.get("monthly_rent") * 12 if item.get("monthly_rent") else None,
                                condition=None,  # Not provided by Propstack
                                features=json.dumps(item.get("amenities", [])) if item.get("amenities") else None,
                                latitude=item["latitude"],
                                longitude=item["longitude"],
                                is_active=item["is_active"],
                                created_at=datetime.utcnow(),
                                updated_at=datetime.utcnow()
                            )
                            
                            session.add(property_obj)
                            session.flush()  # Get the ID
                            property_id = property_obj.id
                            
                            # Add underwriting data if available
                            if any(item.get(field) for field in ["monthly_rent", "cap_rate", "noi"]):
                                underwriting_data = UnderwritingData(
                                    property_id=property_id,
                                    gross_rental_income=item.get("monthly_rent") * 12 if item.get("monthly_rent") else None,
                                    other_income=0,  # Default to 0
                                    total_income=item.get("monthly_rent") * 12 if item.get("monthly_rent") else None,
                                    property_management_fee=None,  # Not provided by Propstack
                                    maintenance_reserves=None,  # Not provided by Propstack
                                    property_taxes=None,  # Not provided by Propstack
                                    insurance=None,  # Not provided by Propstack
                                    utilities=None,  # Not provided by Propstack
                                    other_expenses=None,  # Not provided by Propstack
                                    total_expenses=None,  # Not provided by Propstack
                                    noi=item.get("noi"),
                                    cap_rate=item.get("cap_rate"),
                                    cash_on_cash_return=None,  # Not provided by Propstack
                                    internal_rate_of_return=None,  # Not provided by Propstack
                                    purchase_price=item["price"],
                                    created_at=datetime.utcnow(),
                                    updated_at=datetime.utcnow()
                                )
                                session.add(underwriting_data)
                            
                            saved_count += 1
                            logger.debug(f"Created new property {property_id}")
                        
                    except Exception as e:
                        logger.error(f"Error saving property: {e}")
                        continue
                
                session.commit()
                logger.info(f"Saved {saved_count} new properties and updated {updated_count} existing properties")
                self.stats["successful"] = saved_count + updated_count
                return True
                
        except Exception as e:
            logger.error(f"Database error: {e}")
            return False
    
    def _get_demo_data(self) -> List[Dict[str, Any]]:
        """Get demo data for testing when API key is not available."""
        logger.info("Using demo data for Propstack API")
        
        import random
        
        # Sample property types
        property_types = ["Apartment", "Villa", "Penthouse", "Office", "Retail"]
        
        # Sample cities
        cities = ["New York", "Los Angeles", "Chicago", "Houston", "Miami"]
        
        # Generate demo data
        demo_data = []
        for i in range(5):  # Generate 5 sample properties
            property_type = random.choice(property_types)
            city = random.choice(cities)
            
            # Base price varies by city and property type (in USD)
            base_price = {
                "New York": 1500000,
                "Los Angeles": 1200000,
                "Chicago": 800000,
                "Houston": 600000,
                "Miami": 1000000
            }.get(city, 1000000)
            
            if property_type in ["Office", "Retail"]:
                base_price *= 1.5
            
            # Add some randomness to price
            price = base_price + random.randint(-200000, 200000)
            
            # Calculate monthly rent (approximately 0.5% of property value)
            monthly_rent = price * 0.005
            
            # Calculate cap rate (between 4% and 7%)
            cap_rate = round(random.uniform(4.0, 7.0), 1)
            
            # Calculate NOI
            noi = monthly_rent * 12
            
            # Generate amenities
            amenities = random.sample(["Swimming Pool", "Gym", "Garden", "Clubhouse", "Security", "Power Backup", "Parking", "Lift", "Children's Play Area"], random.randint(3, 7))
            
            # Generate residential-specific fields
            is_residential = property_type not in ["Office", "Retail"]
            bedrooms = random.randint(1, 5) if is_residential else 0
            bathrooms = random.randint(1, 5) if is_residential else 2
            furnishing = random.choice(["Furnished", "Semi-Furnished", "Unfurnished"]) if is_residential else "Bare Shell"
            
            # State mapping
            state_mapping = {
                "New York": "NY",
                "Los Angeles": "CA",
                "Chicago": "IL",
                "Houston": "TX",
                "Miami": "FL"
            }
            
            property_data = {
                "property_id": f"PS-{100000 + i}",
                "property_type": property_type,
                "address": f"{random.randint(1, 100)}, Sample Street, {city}",
                "city": city,
                "state": state_mapping.get(city, "Maharashtra"),
                "pincode": f"{random.randint(100000, 999999)}",
                "price": price,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "carpet_area": random.randint(500, 5000),
                "year_built": random.randint(2000, 2023),
                "latitude": 40.7128 + random.uniform(-0.05, 0.05) if city == "New York" else
                           34.0522 + random.uniform(-0.05, 0.05) if city == "Los Angeles" else
                           41.8781 + random.uniform(-0.05, 0.05) if city == "Chicago" else
                           29.7604 + random.uniform(-0.05, 0.05) if city == "Houston" else
                           25.7617 + random.uniform(-0.05, 0.05),
                "longitude": -74.0060 + random.uniform(-0.05, 0.05) if city == "New York" else
                            -118.2437 + random.uniform(-0.05, 0.05) if city == "Los Angeles" else
                            -87.6298 + random.uniform(-0.05, 0.05) if city == "Chicago" else
                            -95.3698 + random.uniform(-0.05, 0.05) if city == "Houston" else
                            -80.1918 + random.uniform(-0.05, 0.05),
                "description": f"Beautiful {property_type.lower()} in {city} with modern amenities.",
                "listing_date": datetime.utcnow().strftime("%Y-%m-%d"),
                "monthly_rent": monthly_rent,
                "cap_rate": cap_rate,
                "noi": noi,
                "furnishing": furnishing,
                "facing": random.choice(["North", "South", "East", "West"]),
                "floor": random.randint(1, 20),
                "total_floors": random.randint(20, 40),
                "amenities": amenities
            }
            
            demo_data.append(property_data)
        
        return demo_data