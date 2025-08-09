#!/usr/bin/env python3
"""
Sample data generation script for the Real Estate Investment Decision System.
Populates the database with realistic sample data for testing and demonstration.
"""

import sys
import os
import json
from pathlib import Path
import random
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from config.settings import settings
from database.connection import db_manager, init_database
from database.models import Property, TaxRecord, ZoningData, UnderwritingData, MLScore
from underwriting.calculator import UnderwritingCalculator
from ml_models.deal_scorer import DealScorer


def generate_sample_properties():
    """Generate sample property data."""
    
    # Sample cities and states - Indian states and cities
    cities_data = [
        ('Mumbai', 'Maharashtra'),
        ('Delhi', 'Delhi'),
        ('Bangalore', 'Karnataka'),
        ('Hyderabad', 'Telangana'),
        ('Chennai', 'Tamil Nadu'),
        ('Kolkata', 'West Bengal'),
        ('Pune', 'Maharashtra'),
        ('Ahmedabad', 'Gujarat'),
        ('Jaipur', 'Rajasthan'),
        ('Lucknow', 'Uttar Pradesh'),
        ('Chandigarh', 'Punjab'),
        ('Kochi', 'Kerala'),
        ('Guwahati', 'Assam'),
        ('Bhubaneswar', 'Odisha'),
        ('Dehradun', 'Uttarakhand'),
    ]
    
    # Property types
    property_types = ['Single Family', 'Multi-Family', 'Townhouse', 'Condo']
    
    # Street names - Indian style
    street_names = [
        'MG Road', 'Nehru Street', 'Gandhi Marg', 'Patel Road', 'Subhash Chowk',
        'Rajiv Nagar', 'Shastri Avenue', 'Tagore Lane', 'Bose Road', 'Ambedkar Street',
        'Tilak Nagar', 'Azad Marg', 'Sarojini Road', 'Netaji Street', 'Bhagat Singh Lane'
    ]
    
    properties = []
    
    for i in range(100):  # Generate 100 sample properties
        city, state = random.choice(cities_data)
        street_num = random.randint(100, 9999)
        street_name = random.choice(street_names)
        
        # Generate realistic property data
        property_data = {
            'address': f'{street_num} {street_name}',
            'city': city,
            'state': state,
            'zip_code': f'{random.randint(100001, 999999)}',  # Indian PIN codes are 6 digits
            'property_type': random.choice(property_types),
            'bedrooms': random.randint(1, 5),
            'bathrooms': round(random.uniform(1, 4), 1),
            'square_feet': random.randint(800, 3000),
            'year_built': random.randint(1980, 2023),
            'lot_size': round(random.uniform(0.1, 2.0), 2),
            'condition': random.choice(['Excellent', 'Good', 'Fair', 'Needs Work']),
            'features': json.dumps(random.sample(['Parking', 'Gym', 'Power Backup', 'Lift', 'Security', 'Club House', 'Swimming Pool', 'Modular Kitchen'], 
                                    random.randint(0, 4))),
            'latitude': random.uniform(8.0, 35.0),  # India's latitude range
            'longitude': random.uniform(68.0, 97.0),  # India's longitude range
            'source': 'Sample Data',
            'is_active': True
        }
        
        # Generate realistic pricing based on location and property characteristics for Indian market
        # Using INR pricing (approximately 75 INR = 1 USD)
        base_price = property_data['square_feet'] * random.uniform(5000, 12000)
        if property_data['property_type'] == 'Multi-Family':
            base_price *= 1.2
        elif property_data['property_type'] == 'Condo':
            base_price *= 0.9
        
        # Adjust for Indian cities based on real estate market data
        city_multipliers = {
            'Mumbai': 1.8, 'Delhi': 1.5, 'Bangalore': 1.4, 'Hyderabad': 1.2,
            'Chennai': 1.3, 'Kolkata': 1.0, 'Pune': 1.2, 'Ahmedabad': 0.9,
            'Jaipur': 0.8, 'Lucknow': 0.7, 'Chandigarh': 1.0, 'Kochi': 1.1,
            'Guwahati': 0.7, 'Bhubaneswar': 0.6, 'Dehradun': 0.8
        }
        base_price *= city_multipliers.get(city, 1.0)
        
        property_data['list_price'] = round(base_price, -3)  # Round to nearest thousand
        
        # Generate rental data
        rent_multiplier = random.uniform(0.008, 0.012)  # 0.8% to 1.2% of property value
        property_data['monthly_rent'] = round(property_data['list_price'] * rent_multiplier, -10)
        property_data['annual_rent'] = property_data['monthly_rent'] * 12
        
        properties.append(property_data)
    
    return properties


def create_sample_tax_records(property_id):
    """Create sample tax record for a property."""
    assessed_value = random.uniform(0.8, 1.2)  # 80% to 120% of list price
    land_value = assessed_value * random.uniform(0.2, 0.4)
    improvement_value = assessed_value - land_value
    
    return {
        'property_id': property_id,
        'assessed_value': assessed_value,
        'land_value': land_value,
        'improvement_value': improvement_value,
        'annual_taxes': assessed_value * random.uniform(0.015, 0.025),  # 1.5% to 2.5%
        'tax_rate': random.uniform(0.015, 0.025),
        'assessment_year': random.randint(2020, 2024),
        'source': 'Sample Data'
    }


def create_sample_zoning_data(property_id):
    """Create sample zoning data for a property."""
    zoning_codes = ['R1', 'R2', 'R3', 'R4', 'RM1', 'RM2']
    
    return {
        'property_id': property_id,
        'zoning_code': random.choice(zoning_codes),
        'zoning_description': 'Residential Zoning',
        'land_use': 'Residential',
        'max_density': random.uniform(4, 12),
        'max_height': random.uniform(25, 45),
        'setback_requirements': json.dumps({
            'front': random.uniform(15, 25),
            'side': random.uniform(5, 10),
            'rear': random.uniform(15, 25)
        }),
        'permitted_uses': json.dumps(['Single Family Residential', 'Accessory Structures']),
        'conditional_uses': json.dumps(['Home Office', 'Day Care']),
        'source': 'Sample Data'
    }


def main():
    """Main function to generate and populate sample data."""
    logger.info("🏗️ Starting sample data generation...")
    
    try:
        # Initialize database
        init_database()
        logger.info("✅ Database initialized")
        
        # Generate sample properties
        sample_properties = generate_sample_properties()
        logger.info(f"Generated {len(sample_properties)} sample properties")
        
        # Insert properties into database
        with db_manager.get_session() as session:
            property_objects = []
            
            for prop_data in sample_properties:
                property_obj = Property(**prop_data)
                session.add(property_obj)
                property_objects.append(property_obj)
            
            session.commit()
            logger.info(f"✅ Inserted {len(property_objects)} properties")
            
            # Create tax records and zoning data
            for property_obj in property_objects:
                # Tax record
                tax_data = create_sample_tax_records(property_obj.id)
                tax_record = TaxRecord(**tax_data)
                session.add(tax_record)
                
                # Zoning data
                zoning_data = create_sample_zoning_data(property_obj.id)
                zoning_record = ZoningData(**zoning_data)
                session.add(zoning_record)
            
            session.commit()
            logger.info("✅ Created tax records and zoning data")
            
            # Calculate underwriting for all properties
            logger.info("Calculating underwriting metrics...")
            calculator = UnderwritingCalculator()
            
            for property_obj in property_objects:
                try:
                    calculator.calculate_all_metrics(property_obj.id)
                except Exception as e:
                    logger.warning(f"Failed to calculate underwriting for property {property_obj.id}: {e}")
            
            logger.info("✅ Calculated underwriting metrics")
            
            # Train ML model and score properties
            logger.info("Training ML model...")
            scorer = DealScorer()
            
            try:
                X, y = scorer.prepare_training_data()
                if len(X) >= 10:
                    training_results = scorer.train_model(X, y)
                    logger.info(f"✅ ML model trained. Test R²: {training_results['test_score']:.3f}")
                    
                    # Score all properties
                    scoring_results = scorer.score_all_properties()
                    logger.info(f"✅ Scored {len(scoring_results)} properties")
                else:
                    logger.warning("Not enough data for ML training")
            except Exception as e:
                logger.error(f"ML training failed: {e}")
        
        logger.info("🎉 Sample data generation completed successfully!")
        
        # Print summary
        with db_manager.get_session() as session:
            property_count = session.query(Property).count()
            underwriting_count = session.query(UnderwritingData).count()
            ml_score_count = session.query(MLScore).count()
            
            logger.info(f"📊 Database Summary:")
            logger.info(f"   Properties: {property_count}")
            logger.info(f"   Underwriting records: {underwriting_count}")
            logger.info(f"   ML scores: {ml_score_count}")
        
    except Exception as e:
        logger.error(f"❌ Sample data generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
