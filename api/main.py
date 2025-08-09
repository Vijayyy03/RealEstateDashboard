"""
FastAPI backend for the Real Estate Investment Decision System.
Provides REST API endpoints for property data, underwriting calculations, and ML scoring.
"""

import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import uvicorn

from config.settings import settings
from database.connection import get_db, db_manager
from database.models import Property, UnderwritingData, MLScore, TaxRecord, ZoningData
from underwriting.calculator import UnderwritingCalculator
from ml_models.deal_scorer import DealScorer


# Pydantic models for API requests/responses
class PropertyBase(BaseModel):
    address: str
    city: str
    state: str
    zip_code: str
    property_type: Optional[str] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    square_feet: Optional[int] = None
    year_built: Optional[int] = None
    list_price: Optional[float] = None
    monthly_rent: Optional[float] = None
    annual_rent: Optional[float] = None


class PropertyCreate(PropertyBase):
    pass

class PropertyResponse(PropertyBase):
    id: int
    mls_id: Optional[str] = None
    county: Optional[str] = None
    lot_size: Optional[float] = None
    condition: Optional[str] = None
    features: Optional[List[str]] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    source: str
    created_at: datetime
    updated_at: datetime
    is_active: bool

    class Config:
        from_attributes = True

    pass


class UnderwritingResponse(BaseModel):
    property_id: int
    gross_rental_income: Optional[float] = None
    other_income: Optional[float] = None
    total_income: Optional[float] = None
    property_management_fee: Optional[float] = None
    maintenance_reserves: Optional[float] = None
    property_taxes: Optional[float] = None
    insurance: Optional[float] = None
    utilities: Optional[float] = None
    other_expenses: Optional[float] = None
    total_expenses: Optional[float] = None
    noi: Optional[float] = None
    cap_rate: Optional[float] = None
    cash_on_cash_return: Optional[float] = None
    internal_rate_of_return: Optional[float] = None
    purchase_price: Optional[float] = None
    down_payment: Optional[float] = None
    loan_amount: Optional[float] = None
    interest_rate: Optional[float] = None
    loan_term: Optional[int] = None
    principal_interest: Optional[float] = None
    property_tax_payment: Optional[float] = None
    insurance_payment: Optional[float] = None
    total_monthly_payment: Optional[float] = None
    monthly_cash_flow: Optional[float] = None
    annual_cash_flow: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MLScoreResponse(BaseModel):
    property_id: int
    model_name: str
    model_version: str
    deal_score: Optional[float] = None
    risk_score: Optional[float] = None
    yield_prediction: Optional[float] = None
    price_prediction: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    confidence_score: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

    pass


class InvestmentReport(BaseModel):
    property_info: Dict[str, Any]
    financial_metrics: Dict[str, Any]
    investment_grade: str
    risk_assessment: Dict[str, Any]
    recommendations: List[str]


class PortfolioMetrics(BaseModel):
    total_properties: int
    total_investment: float
    total_noi: float
    total_cash_flow: float
    weighted_cap_rate: float
    average_cash_on_cash: float
    properties: List[Dict[str, Any]]


def convert_property_for_response(property_obj):
    """Convert property object for API response, handling JSON strings."""
    # Create a copy of the object to avoid modifying the database object
    from copy import copy
    obj_copy = copy(property_obj)
    
    if hasattr(obj_copy, 'features') and obj_copy.features and isinstance(obj_copy.features, str):
        try:
            obj_copy.features = json.loads(obj_copy.features)
        except (json.JSONDecodeError, TypeError):
            obj_copy.features = []
    return obj_copy

def convert_ml_score_for_response(ml_score_obj):
    """Convert ML score object for API response, handling JSON strings."""
    # Create a copy of the object to avoid modifying the database object
    from copy import copy
    obj_copy = copy(ml_score_obj)
    
    if hasattr(obj_copy, 'feature_importance') and obj_copy.feature_importance and isinstance(obj_copy.feature_importance, str):
        try:
            obj_copy.feature_importance = json.loads(obj_copy.feature_importance)
        except (json.JSONDecodeError, TypeError):
            obj_copy.feature_importance = {}
    return obj_copy

# Initialize FastAPI app
app = FastAPI(
    title=settings.api.title,
    version=settings.api.version,
    description="AI-powered Real Estate Investment Decision System API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.api.version
    }


# Property endpoints
@app.get("/properties", response_model=List[PropertyResponse])
async def get_properties(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    city: Optional[str] = None,
    state: Optional[str] = None,
    property_type: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """Get properties with optional filtering."""
    query = db.query(Property).filter(Property.is_active == True)
    
    if city:
        query = query.filter(Property.city.ilike(f"%{city}%"))
    
    if state:
        query = query.filter(Property.state == state.upper())
    
    if property_type:
        query = query.filter(Property.property_type == property_type)
    
    if min_price is not None:
        query = query.filter(Property.list_price >= min_price)
    
    if max_price is not None:
        query = query.filter(Property.list_price <= max_price)
    
    properties = query.offset(skip).limit(limit).all()
    # Convert JSON strings to proper types
    converted_properties = [convert_property_for_response(prop) for prop in properties]
    return converted_properties


@app.get("/properties/{property_id}", response_model=PropertyResponse)
async def get_property(property_id: int, db: Session = Depends(get_db)):
    """Get a specific property by ID."""
    property_obj = db.query(Property).filter(Property.id == property_id).first()
    
    if not property_obj:
        raise HTTPException(status_code=404, detail="Property not found")
    
    # Convert JSON strings to proper types
    converted_property = convert_property_for_response(property_obj)
    return converted_property


@app.post("/properties", response_model=PropertyResponse)
async def create_property(property_data: PropertyCreate, db: Session = Depends(get_db)):
    """Create a new property."""
    property_obj = Property(**property_data.dict())
    db.add(property_obj)
    db.commit()
    db.refresh(property_obj)
    return property_obj


# Underwriting endpoints
@app.get("/properties/{property_id}/underwriting", response_model=UnderwritingResponse)
async def get_underwriting(property_id: int, db: Session = Depends(get_db)):
    """Get underwriting data for a property."""
    underwriting_data = db.query(UnderwritingData).filter(
        UnderwritingData.property_id == property_id
    ).first()
    
    if not underwriting_data:
        raise HTTPException(status_code=404, detail="Underwriting data not found")
    
    return underwriting_data


@app.post("/properties/{property_id}/underwriting")
async def calculate_underwriting(
    property_id: int,
    custom_assumptions: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
):
    """Calculate underwriting metrics for a property."""
    try:
        calculator = UnderwritingCalculator()
        results = calculator.calculate_all_metrics(property_id, custom_assumptions)
        
        return {
            "property_id": property_id,
            "underwriting_data": results,
            "calculated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/properties/{property_id}/investment-report", response_model=InvestmentReport)
async def get_investment_report(property_id: int, db: Session = Depends(get_db)):
    """Get comprehensive investment report for a property."""
    try:
        calculator = UnderwritingCalculator()
        report = calculator.generate_investment_report(property_id)
        return report
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ML Scoring endpoints
@app.get("/properties/{property_id}/ml-score", response_model=MLScoreResponse)
async def get_ml_score(property_id: int, db: Session = Depends(get_db)):
    """Get ML score for a property."""
    ml_score = db.query(MLScore).filter(
        MLScore.property_id == property_id,
        MLScore.model_name == settings.ml.scoring_model_name
    ).first()
    
    if not ml_score:
        raise HTTPException(status_code=404, detail="ML score not found")
    
    return ml_score


@app.post("/properties/{property_id}/ml-score")
async def calculate_ml_score(property_id: int, db: Session = Depends(get_db)):
    """Calculate ML score for a property."""
    try:
        scorer = DealScorer()
        prediction = scorer.predict_deal_score(property_id)
        
        return {
            "property_id": property_id,
            "prediction": prediction,
            "calculated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ml/train")
async def train_ml_model():
    """Train the ML model."""
    try:
        scorer = DealScorer()
        X, y = scorer.prepare_training_data()
        
        if len(X) < 10:
            raise HTTPException(
                status_code=400,
                detail="Not enough data for training (need at least 10 samples)"
            )
        
        results = scorer.train_model(X, y)
        
        return {
            "status": "success",
            "training_results": results,
            "trained_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ml/score-all")
async def score_all_properties():
    """Score all properties in the database."""
    try:
        scorer = DealScorer()
        results = scorer.score_all_properties()
        
        return {
            "status": "success",
            "properties_scored": len(results),
            "results": results,
            "scored_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Portfolio endpoints
@app.post("/portfolio/calculate", response_model=PortfolioMetrics)
async def calculate_portfolio_metrics(property_ids: List[int]):
    """Calculate portfolio metrics for multiple properties."""
    try:
        calculator = UnderwritingCalculator()
        portfolio_results = calculator.calculate_portfolio_metrics(property_ids)
        return portfolio_results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Search and analytics endpoints
@app.get("/search/properties")
async def search_properties(
    q: str = Query(..., min_length=1),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Search properties by address, city, or other criteria."""
    query = db.query(Property).filter(
        Property.is_active == True,
        (Property.address.ilike(f"%{q}%")) |
        (Property.city.ilike(f"%{q}%")) |
        (Property.state.ilike(f"%{q}%"))
    )
    
    properties = query.limit(limit).all()
    return properties


@app.get("/analytics/market-summary")
async def get_market_summary(db: Session = Depends(get_db)):
    """Get market summary statistics."""
    try:
        # Property count by state
        state_counts = db.query(
            Property.state,
            db.func.count(Property.id).label('count')
        ).filter(
            Property.is_active == True
        ).group_by(Property.state).all()
        
        # Average prices by property type
        avg_prices = db.query(
            Property.property_type,
            db.func.avg(Property.list_price).label('avg_price')
        ).filter(
            Property.is_active == True,
            Property.list_price.isnot(None)
        ).group_by(Property.property_type).all()
        
        # Cap rate statistics
        cap_rate_stats = db.query(
            db.func.avg(UnderwritingData.cap_rate).label('avg_cap_rate'),
            db.func.min(UnderwritingData.cap_rate).label('min_cap_rate'),
            db.func.max(UnderwritingData.cap_rate).label('max_cap_rate')
        ).first()
        
        return {
            "property_count_by_state": [{"state": s.state, "count": s.count} for s in state_counts],
            "avg_prices_by_type": [{"type": p.property_type, "avg_price": float(p.avg_price)} for p in avg_prices],
            "cap_rate_stats": {
                "average": float(cap_rate_stats.avg_cap_rate) if cap_rate_stats.avg_cap_rate else None,
                "minimum": float(cap_rate_stats.min_cap_rate) if cap_rate_stats.min_cap_rate else None,
                "maximum": float(cap_rate_stats.max_cap_rate) if cap_rate_stats.max_cap_rate else None
            },
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/analytics/top-deals")
async def get_top_deals(
    limit: int = Query(10, ge=1, le=100),
    min_score: float = Query(0, ge=0, le=100),
    db: Session = Depends(get_db)
):
    """Get top investment deals based on ML scores."""
    try:
        top_deals = db.query(
            Property, MLScore
        ).join(
            MLScore, Property.id == MLScore.property_id
        ).filter(
            Property.is_active == True,
            MLScore.model_name == settings.ml.scoring_model_name,
            MLScore.deal_score >= min_score
        ).order_by(
            MLScore.deal_score.desc()
        ).limit(limit).all()
        
        results = []
        for property_obj, ml_score in top_deals:
            results.append({
                "property": {
                    "id": property_obj.id,
                    "address": property_obj.address,
                    "city": property_obj.city,
                    "state": property_obj.state,
                    "property_type": property_obj.property_type,
                    "list_price": property_obj.list_price,
                    "bedrooms": property_obj.bedrooms,
                    "bathrooms": property_obj.bathrooms,
                    "square_feet": property_obj.square_feet
                },
                "ml_score": {
                    "deal_score": ml_score.deal_score,
                    "risk_score": ml_score.risk_score,
                    "confidence": ml_score.confidence_score
                }
            })
        
        return {
            "top_deals": results,
            "count": len(results),
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug
    )
