"""
Database models for the Real Estate Investment Decision System.
Uses SQLAlchemy with SQLite for demo purposes.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import json

Base = declarative_base()


class Property(Base):
    """Property model for storing real estate listings and data."""
    
    __tablename__ = "properties"
    
    id = Column(Integer, primary_key=True, index=True)
    mls_id = Column(String(50), unique=True, index=True, nullable=True)
    address = Column(String(255), nullable=False)
    city = Column(String(100), nullable=False)
    state = Column(String(2), nullable=False)
    zip_code = Column(String(10), nullable=False)
    county = Column(String(100), nullable=True)
    
    # Property details
    property_type = Column(String(50), nullable=True)  # Single Family, Multi-Family, Commercial, etc.
    bedrooms = Column(Integer, nullable=True)
    bathrooms = Column(Float, nullable=True)
    square_feet = Column(Integer, nullable=True)
    lot_size = Column(Float, nullable=True)  # in acres
    year_built = Column(Integer, nullable=True)
    
    # Financial data
    list_price = Column(Float, nullable=True)
    sold_price = Column(Float, nullable=True)
    estimated_value = Column(Float, nullable=True)
    
    # Rental data
    monthly_rent = Column(Float, nullable=True)
    annual_rent = Column(Float, nullable=True)
    
    # Property condition
    condition = Column(String(50), nullable=True)
    features = Column(Text, nullable=True)  # Store as JSON string for SQLite
    
    # Spatial data
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    # Metadata
    source = Column(String(50), nullable=False)  # MLS, API, Manual, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    tax_records = relationship("TaxRecord", back_populates="property")
    zoning_data = relationship("ZoningData", back_populates="property")
    underwriting_data = relationship("UnderwritingData", back_populates="property")
    ml_scores = relationship("MLScore", back_populates="property")
    
    # Indexes
    __table_args__ = (
        Index('idx_properties_location', 'city', 'state'),
        Index('idx_properties_price', 'list_price'),
        Index('idx_properties_type', 'property_type'),
    )


class TaxRecord(Base):
    """Tax assessment and property tax records."""
    
    __tablename__ = "tax_records"
    
    id = Column(Integer, primary_key=True, index=True)
    property_id = Column(Integer, ForeignKey("properties.id"), nullable=False)
    
    # Assessment data
    assessed_value = Column(Float, nullable=True)
    land_value = Column(Float, nullable=True)
    improvement_value = Column(Float, nullable=True)
    
    # Tax data
    annual_taxes = Column(Float, nullable=True)
    tax_rate = Column(Float, nullable=True)  # percentage
    
    # Assessment year
    assessment_year = Column(Integer, nullable=True)
    
    # Metadata
    source = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    property = relationship("Property", back_populates="tax_records")


class ZoningData(Base):
    """Zoning and land use information."""
    
    __tablename__ = "zoning_data"
    
    id = Column(Integer, primary_key=True, index=True)
    property_id = Column(Integer, ForeignKey("properties.id"), nullable=False)
    
    # Zoning information
    zoning_code = Column(String(50), nullable=True)
    zoning_description = Column(Text, nullable=True)
    land_use = Column(String(100), nullable=True)
    
    # Development potential
    max_density = Column(Float, nullable=True)  # units per acre
    max_height = Column(Float, nullable=True)  # feet
    setback_requirements = Column(Text, nullable=True)  # Store as JSON string
    
    # Permitted uses
    permitted_uses = Column(Text, nullable=True)  # Store as JSON string
    conditional_uses = Column(Text, nullable=True)  # Store as JSON string
    
    # Metadata
    source = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    property = relationship("Property", back_populates="zoning_data")


class UnderwritingData(Base):
    """Financial underwriting calculations and analysis."""
    
    __tablename__ = "underwriting_data"
    
    id = Column(Integer, primary_key=True, index=True)
    property_id = Column(Integer, ForeignKey("properties.id"), nullable=False)
    
    # Income analysis
    gross_rental_income = Column(Float, nullable=True)
    other_income = Column(Float, nullable=True)
    total_income = Column(Float, nullable=True)
    
    # Expense analysis
    property_management_fee = Column(Float, nullable=True)
    maintenance_reserves = Column(Float, nullable=True)
    property_taxes = Column(Float, nullable=True)
    insurance = Column(Float, nullable=True)
    utilities = Column(Float, nullable=True)
    other_expenses = Column(Float, nullable=True)
    total_expenses = Column(Float, nullable=True)
    
    # Key metrics
    noi = Column(Float, nullable=True)  # Net Operating Income
    cap_rate = Column(Float, nullable=True)  # Capitalization Rate
    cash_on_cash_return = Column(Float, nullable=True)
    internal_rate_of_return = Column(Float, nullable=True)
    
    # Purchase assumptions
    purchase_price = Column(Float, nullable=True)
    down_payment = Column(Float, nullable=True)
    loan_amount = Column(Float, nullable=True)
    interest_rate = Column(Float, nullable=True)
    loan_term = Column(Integer, nullable=True)  # years
    
    # Monthly payments
    principal_interest = Column(Float, nullable=True)
    property_tax_payment = Column(Float, nullable=True)
    insurance_payment = Column(Float, nullable=True)
    total_monthly_payment = Column(Float, nullable=True)
    
    # Cash flow
    monthly_cash_flow = Column(Float, nullable=True)
    annual_cash_flow = Column(Float, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    property = relationship("Property", back_populates="underwriting_data")


class MLScore(Base):
    """Machine learning model scores and predictions."""
    
    __tablename__ = "ml_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    property_id = Column(Integer, ForeignKey("properties.id"), nullable=False)
    
    # Model information
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20), nullable=False)
    
    # Scores and predictions
    deal_score = Column(Float, nullable=True)  # 0-100 scale
    risk_score = Column(Float, nullable=True)  # 0-100 scale
    yield_prediction = Column(Float, nullable=True)  # predicted annual return
    price_prediction = Column(Float, nullable=True)  # predicted property value
    
    # Feature importance
    feature_importance = Column(Text, nullable=True)  # Store as JSON string
    
    # Model confidence
    confidence_score = Column(Float, nullable=True)  # 0-1 scale
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    property = relationship("Property", back_populates="ml_scores")


class MarketData(Base):
    """Market-level data and trends."""
    
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Location
    city = Column(String(100), nullable=False)
    state = Column(String(2), nullable=False)
    zip_code = Column(String(10), nullable=True)
    
    # Market metrics
    median_price = Column(Float, nullable=True)
    median_rent = Column(Float, nullable=True)
    price_per_sqft = Column(Float, nullable=True)
    rent_per_sqft = Column(Float, nullable=True)
    
    # Market trends
    price_growth_rate = Column(Float, nullable=True)  # annual percentage
    rent_growth_rate = Column(Float, nullable=True)  # annual percentage
    days_on_market = Column(Float, nullable=True)
    
    # Economic indicators
    unemployment_rate = Column(Float, nullable=True)
    population_growth = Column(Float, nullable=True)
    job_growth = Column(Float, nullable=True)
    
    # Date
    data_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_market_data_location', 'city', 'state'),
        Index('idx_market_data_date', 'data_date'),
    )


class DataIngestionLog(Base):
    """Log of data ingestion activities."""
    
    __tablename__ = "data_ingestion_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Ingestion details
    source = Column(String(50), nullable=False)
    ingestion_type = Column(String(50), nullable=False)  # MLS, Tax, Zoning, etc.
    
    # Results
    records_processed = Column(Integer, nullable=False)
    records_successful = Column(Integer, nullable=False)
    records_failed = Column(Integer, nullable=False)
    
    # Timing
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    duration_seconds = Column(Float, nullable=True)
    
    # Status
    status = Column(String(20), nullable=False)  # SUCCESS, FAILED, PARTIAL
    error_message = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_ingestion_log_source', 'source'),
        Index('idx_ingestion_log_date', 'created_at'),
    )
