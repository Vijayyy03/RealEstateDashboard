# Propstack API Integration

This document describes the integration of the Propstack API for US real estate data into the Real Estate Investment Decision System.

## Overview

Propstack is a leading real estate data, analytics, and workflow solutions platform. It provides comprehensive data for various real estate sectors including office, residential, and warehousing. This integration allows the system to fetch property data from the US market.

## Configuration

### API Settings

The Propstack API integration requires the following configuration in your `.env` file:

```
PROPSTACK_API_KEY=your_propstack_api_key_here
PROPSTACK_BASE_URL=https://api.propstack.in/v1
API_DATA_ENABLED=true
```

- `PROPSTACK_API_KEY`: Your Propstack API key
- `PROPSTACK_BASE_URL`: The base URL for the Propstack API
- `API_DATA_ENABLED`: Enable/disable API data collection

## Implementation

The Propstack API integration is implemented in the following files:

- `data_ingestion/propstack_scraper.py`: The main scraper implementation
- `config/settings.py`: Configuration settings for the Propstack API
- `data_ingestion/main.py`: Integration with the data ingestion pipeline

### PropstackScraper

The `PropstackScraper` class extends the `BaseScraper` class and implements the following methods:

- `get_required_fields()`: Defines the required fields for property data
- `scrape_data(**kwargs)`: Fetches data from the Propstack API
- `process_data(raw_data)`: Processes and standardizes the raw data
- `save_data(processed_data)`: Saves the processed data to the database

### Demo Data

If the Propstack API key is not configured, the scraper will use demo data that represents properties in major US cities (New York, Los Angeles, Chicago, Houston, and Miami).

## Usage

### Running the Scraper

You can run the Propstack API scraper using the following command:

```bash
python data_ingestion/main.py --mode api
```

### Testing the Scraper

A test script is provided to verify the PropstackScraper implementation:

```bash
python scripts/test_propstack_scraper.py
```

## Data Fields

The Propstack API provides the following data fields for properties:

- Basic Information: property_id, property_type, address, city, state, pincode
- Property Details: price, bedrooms, bathrooms, carpet_area, year_built
- Location: latitude, longitude
- Financial Data: monthly_rent, cap_rate, noi
- Additional Fields: furnishing, facing, floor, total_floors, amenities

## Database Integration

### Property Model
The scraper maps Propstack data to the following fields in the Property model:

- source: "Propstack"
- mls_id: property_id
- property_type: property_type
- address: address
- city: city
- state: state
- zip_code: pincode
- list_price: price
- bedrooms: bedrooms
- bathrooms: bathrooms
- square_feet: carpet_area
- year_built: year_built
- latitude: latitude
- longitude: longitude
- features: JSON array containing amenities and additional features (furnishing, facing, floor, total_floors)

### UnderwritingData Model
Financial data is stored in the UnderwritingData model:

- gross_rental_income: monthly_rent * 12
- total_income: monthly_rent * 12
- noi: noi
- cap_rate: cap_rate
- purchase_price: price

## Integration with Full Pipeline

The Propstack API data collection is integrated into the full data ingestion pipeline. When running the full pipeline, the API data collection will be executed if enabled in the settings.

```bash
python data_ingestion/main.py --mode full
```

## Error Handling

The PropstackScraper includes error handling for API requests, data processing, and database operations. Errors are logged using the loguru logger and can be found in the application log file.