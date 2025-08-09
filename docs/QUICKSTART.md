# Quick Start Guide - Real Estate Investment Decision System

This guide will help you get the Real Estate Investment Decision System up and running quickly.

## 🚀 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.9+** - [Download Python](https://www.python.org/downloads/)
- **PostgreSQL 12+** with PostGIS extension - [Download PostgreSQL](https://www.postgresql.org/download/)
- **Git** - [Download Git](https://git-scm.com/downloads)

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd RealEstate
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Database

#### Install PostgreSQL and PostGIS

**Windows:**
1. Download and install PostgreSQL from the official website
2. During installation, ensure PostGIS extension is selected
3. Note down your database password

**macOS:**
```bash
brew install postgresql postgis
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib postgis
```

#### Create Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE real_estate_db;

# Enable PostGIS extension
\c real_estate_db
CREATE EXTENSION postgis;
CREATE EXTENSION postgis_topology;

# Exit psql
\q
```

### 5. Configure Environment

```bash
# Copy environment template
cp config/env.example config/.env

# Edit the configuration file
# Update database credentials and other settings
```

Edit `config/.env` with your database credentials:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=real_estate_db
DB_USER=postgres
DB_PASSWORD=your_password_here

# Other settings can remain as defaults for now
```

### 6. Initialize Database

```bash
python database/setup_db.py
```

## 🎯 Quick Demo

### 1. Generate Sample Data

```bash
python scripts/sample_data.py
```

This will create 100 sample properties with realistic data for testing.

### 2. Start the Dashboard

```bash
streamlit run dashboard/main.py
```

Open your browser and navigate to `http://localhost:8501`

### 3. Start the API Server (Optional)

```bash
python api/main.py
```

The API will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📊 What You'll See

### Dashboard Features

1. **Property Explorer**
   - View all properties in the database
   - Filter by location, price, property type
   - Sort by deal score, cap rate, etc.

2. **Investment Analysis**
   - Deal rankings based on ML scores
   - Financial metrics visualization
   - Market analysis charts

3. **Property Details**
   - Individual property analysis
   - Underwriting calculations
   - Investment recommendations

### Sample Data Includes

- 100 properties across 10 cities
- Realistic pricing and rental data
- Tax records and zoning information
- Pre-calculated underwriting metrics
- ML deal scores

## 🔧 Running Data Ingestion

### Manual Data Ingestion

```bash
# Run full pipeline
python data_ingestion/main.py --mode full

# Run specific components
python data_ingestion/main.py --mode mls
python data_ingestion/main.py --mode underwriting
python data_ingestion/main.py --mode ml
```

### Available Modes

- `full` - Complete pipeline (default)
- `mls` - MLS data scraping only
- `tax` - Tax records scraping only
- `zoning` - Zoning data scraping only
- `api` - API data collection only
- `underwriting` - Underwriting calculations only
- `ml` - ML model training and scoring only

## 🧪 Testing

### Run Unit Tests

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_underwriting.py

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Test API Endpoints

```bash
# Test health check
curl http://localhost:8000/health

# Get properties
curl http://localhost:8000/properties

# Get top deals
curl http://localhost:8000/analytics/top-deals
```

## 📈 Understanding the System

### Core Components

1. **Data Ingestion** (`data_ingestion/`)
   - Web scraping for property listings
   - API integrations for external data
   - Data cleaning and validation

2. **Underwriting Engine** (`underwriting/`)
   - Financial calculations (NOI, Cap Rate, etc.)
   - Investment analysis
   - Risk assessment

3. **ML Models** (`ml_models/`)
   - Deal scoring algorithms
   - Predictive analytics
   - Model training and evaluation

4. **Dashboard** (`dashboard/`)
   - Interactive web interface
   - Data visualization
   - Property filtering and search

5. **API** (`api/`)
   - REST API endpoints
   - Data access and manipulation
   - Integration capabilities

### Key Metrics

- **Cap Rate**: Net Operating Income / Property Value
- **Cash-on-Cash Return**: Annual Cash Flow / Down Payment
- **Deal Score**: ML-based investment score (0-100)
- **NOI**: Net Operating Income (Gross Income - Expenses)

## 🚨 Troubleshooting

### Common Issues

**Database Connection Error**
```bash
# Check if PostgreSQL is running
# Windows: Check Services
# macOS: brew services start postgresql
# Linux: sudo systemctl start postgresql

# Test connection
psql -U postgres -d real_estate_db
```

**Missing Dependencies**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**Port Already in Use**
```bash
# Check what's using the port
# Windows: netstat -ano | findstr :8501
# macOS/Linux: lsof -i :8501

# Kill the process or use different port
streamlit run dashboard/main.py --server.port 8502
```

**Permission Errors**
```bash
# Ensure you have write permissions
# Windows: Run as Administrator
# macOS/Linux: Check file permissions
chmod +x scripts/sample_data.py
```

### Getting Help

1. Check the logs in `logs/app.log`
2. Review the configuration in `config/.env`
3. Ensure all prerequisites are installed
4. Verify database connectivity

## 🎯 Next Steps

After getting the system running:

1. **Explore the Dashboard**
   - Try different filters and views
   - Analyze sample properties
   - Understand the metrics

2. **Test the API**
   - Use the interactive docs at `/docs`
   - Try different endpoints
   - Understand the data structure

3. **Add Your Own Data**
   - Modify the sample data script
   - Add real property data
   - Test with your own scenarios

4. **Customize the System**
   - Modify underwriting assumptions
   - Adjust ML model parameters
   - Add new data sources

## 📚 Additional Resources

- [Project README](README.md) - Complete project overview
- [Development Roadmap](ROADMAP.md) - Future development plans
- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [Configuration Guide](config/) - Detailed configuration options

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the logs for error messages
3. Ensure all prerequisites are met
4. Try the sample data generation first

For additional help, please refer to the project documentation or create an issue in the repository.

---

**Happy Investing! 🏠📈**
