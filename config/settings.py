"""
Configuration settings for the Real Estate Investment Decision System.
Uses Pydantic for type-safe configuration management.
"""

import os
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    name: str = Field(default="real_estate_db", env="DB_NAME")
    user: str = Field(default="postgres", env="DB_USER")
    password: str = Field(default="", env="DB_PASSWORD")
    
    @property
    def url(self) -> str:
        """Generate database URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class APISettings(BaseSettings):
    """API configuration settings."""
    
    title: str = Field(default="Real Estate Investment API", env="API_TITLE")
    version: str = Field(default="1.0.0", env="API_VERSION")
    debug: bool = Field(default=False, env="API_DEBUG")
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    cors_origins: List[str] = Field(default=["http://localhost:3000"], env="CORS_ORIGINS")


class ScrapingSettings(BaseSettings):
    """Web scraping configuration settings."""
    
    # General scraping settings
    request_delay: float = Field(default=1.0, env="SCRAPING_DELAY")
    max_retries: int = Field(default=3, env="SCRAPING_MAX_RETRIES")
    timeout: int = Field(default=30, env="SCRAPING_TIMEOUT")
    user_agent: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        env="SCRAPING_USER_AGENT"
    )
    
    # MLS scraping settings
    mls_enabled: bool = Field(default=True, env="MLS_SCRAPING_ENABLED")
    mls_batch_size: int = Field(default=100, env="MLS_BATCH_SIZE")
    
    # Tax records settings
    tax_records_enabled: bool = Field(default=True, env="TAX_RECORDS_ENABLED")
    
    # Zoning data settings
    zoning_enabled: bool = Field(default=True, env="ZONING_ENABLED")
    
    # API data collection settings
    api_enabled: bool = Field(default=True, env="API_DATA_ENABLED")


class MLSettings(BaseSettings):
    """Machine Learning model configuration settings."""
    
    # Model paths
    model_dir: str = Field(default="ml_models/saved_models", env="ML_MODEL_DIR")
    
    # Scoring model settings
    scoring_model_name: str = Field(default="deal_scorer", env="ML_SCORING_MODEL")
    prediction_model_name: str = Field(default="yield_predictor", env="ML_PREDICTION_MODEL")
    
    # Feature engineering
    feature_columns: List[str] = Field(
        default=[
            "price", "sqft", "bedrooms", "bathrooms", "year_built",
            "cap_rate", "noi", "cash_on_cash", "location_score"
        ],
        env="ML_FEATURE_COLUMNS"
    )
    
    # Model hyperparameters
    random_state: int = Field(default=42, env="ML_RANDOM_STATE")
    test_size: float = Field(default=0.2, env="ML_TEST_SIZE")


class DashboardSettings(BaseSettings):
    """Dashboard configuration settings."""
    
    title: str = Field(default="Real Estate Investment Dashboard", env="DASHBOARD_TITLE")
    theme: str = Field(default="light", env="DASHBOARD_THEME")
    page_icon: str = Field(default="🏠", env="DASHBOARD_ICON")
    layout: str = Field(default="wide", env="DASHBOARD_LAYOUT")


class ExternalAPISettings(BaseSettings):
    """External API configuration settings."""
    
    # Zillow API
    zillow_api_key: Optional[str] = Field(default=None, env="ZILLOW_API_KEY")
    zillow_base_url: str = Field(default="https://api.bridgedataoutput.com", env="ZILLOW_BASE_URL")
    
    # Redfin API
    redfin_api_key: Optional[str] = Field(default=None, env="REDFIN_API_KEY")
    
    # Census API
    census_api_key: Optional[str] = Field(default=None, env="CENSUS_API_KEY")
    
    # Google Maps API
    google_maps_api_key: Optional[str] = Field(default=None, env="GOOGLE_MAPS_API_KEY")
    
    # Propstack API (Indian Real Estate Data)
    propstack_api_key: Optional[str] = Field(default=None, env="PROPSTACK_API_KEY")
    propstack_base_url: str = Field(default="https://api.propstack.in/v1", env="PROPSTACK_BASE_URL")


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", env="LOG_FILE")
    
    # Database
    database: DatabaseSettings = DatabaseSettings()
    
    # API
    api: APISettings = APISettings()
    
    # Scraping
    scraping: ScrapingSettings = ScrapingSettings()
    
    # Machine Learning
    ml: MLSettings = MLSettings()
    
    # Dashboard
    dashboard: DashboardSettings = DashboardSettings()
    
    # External APIs
    external_apis: ExternalAPISettings = ExternalAPISettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
