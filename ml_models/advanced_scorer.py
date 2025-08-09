"""
Advanced ML Model for Real Estate Investment Scoring.
Includes feature engineering, multiple algorithms, and ensemble methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import joblib
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import xgboost as xgb
import lightgbm as lgb
from loguru import logger

from database.connection import db_manager
from database.models import Property, UnderwritingData, MLScore, TaxRecord, ZoningData
from config.settings import settings


class AdvancedDealScorer:
    """Advanced ML model for deal scoring with feature engineering and ensemble methods."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.feature_columns = []
        self.target_column = 'deal_score'
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': 42
                }
            },
            'xgboost': {
                'model': xgb.XGBRegressor,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': 42
                }
            },
            'lightgbm': {
                'model': lgb.LGBMRegressor,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': 42
                }
            }
        }
    
    def prepare_advanced_features(self) -> pd.DataFrame:
        """Prepare advanced features with engineering and selection."""
        logger.info("🔧 Preparing advanced features...")
        
        try:
            with db_manager.get_session() as session:
                # Query all data
                query = session.query(
                    Property, UnderwritingData, MLScore, TaxRecord, ZoningData
                ).outerjoin(
                    UnderwritingData, Property.id == UnderwritingData.property_id
                ).outerjoin(
                    MLScore, Property.id == MLScore.property_id
                ).outerjoin(
                    TaxRecord, Property.id == TaxRecord.property_id
                ).outerjoin(
                    ZoningData, Property.id == ZoningData.property_id
                ).filter(
                    Property.is_active == True
                )
                
                data = []
                for prop, underwriting, ml_score, tax, zoning in query.all():
                    row = self._extract_property_features(prop, underwriting, ml_score, tax, zoning)
                    if row:
                        data.append(row)
                
                df = pd.DataFrame(data)
                
                if df.empty:
                    logger.warning("No data available for feature engineering")
                    return df
                
                # Feature engineering
                df = self._engineer_features(df)
                
                # Feature selection
                df = self._select_features(df)
                
                logger.info(f"✅ Prepared {len(df)} samples with {len(self.feature_columns)} features")
                return df
                
        except Exception as e:
            logger.error(f"❌ Error preparing features: {e}")
            return pd.DataFrame()
    
    def _extract_property_features(self, prop, underwriting, ml_score, tax, zoning) -> Optional[Dict[str, Any]]:
        """Extract features from property and related data."""
        try:
            row = {
                'property_id': prop.id,
                'address': prop.address,
                'city': prop.city,
                'state': prop.state,
                'zip_code': prop.zip_code,
                'property_type': prop.property_type,
                'bedrooms': prop.bedrooms,
                'bathrooms': prop.bathrooms,
                'square_feet': prop.square_feet,
                'year_built': prop.year_built,
                'list_price': prop.list_price,
                'monthly_rent': prop.monthly_rent,
                'annual_rent': prop.annual_rent,
                'condition': prop.condition,
                'features': prop.features,
                'latitude': prop.latitude,
                'longitude': prop.longitude,
                'source': prop.source
            }
            
            # Add underwriting data
            if underwriting:
                underwriting_dict = {
                    'gross_rental_income': underwriting.gross_rental_income,
                    'total_income': underwriting.total_income,
                    'total_expenses': underwriting.total_expenses,
                    'noi': underwriting.noi,
                    'cap_rate': underwriting.cap_rate,
                    'cash_on_cash_return': underwriting.cash_on_cash_return,
                    'internal_rate_of_return': underwriting.internal_rate_of_return,
                    'monthly_cash_flow': underwriting.monthly_cash_flow,
                    'annual_cash_flow': underwriting.annual_cash_flow
                }
                row.update(underwriting_dict)
            
            # Add ML score data
            if ml_score:
                ml_dict = {
                    'deal_score': ml_score.deal_score,
                    'risk_score': ml_score.risk_score,
                    'yield_prediction': ml_score.yield_prediction,
                    'price_prediction': ml_score.price_prediction,
                    'confidence_score': ml_score.confidence_score
                }
                row.update(ml_dict)
            else:
                # If no ML score exists, create default values
                ml_dict = {
                    'deal_score': 50.0,  # Default deal score
                    'risk_score': 50.0,  # Default risk score
                    'yield_prediction': 0.0,
                    'price_prediction': 0.0,
                    'confidence_score': 0.0
                }
                row.update(ml_dict)
            
            # Add tax data
            if tax:
                tax_dict = {
                    'assessed_value': tax.assessed_value,
                    'land_value': tax.land_value,
                    'improvement_value': tax.improvement_value,
                    'annual_taxes': tax.annual_taxes,
                    'tax_rate': tax.tax_rate,
                    'assessment_year': tax.assessment_year
                }
                row.update(tax_dict)
            
            # Add zoning data
            if zoning:
                zoning_dict = {
                    'zoning_code': zoning.zoning_code,
                    'land_use': zoning.land_use,
                    'max_density': zoning.max_density,
                    'max_height': zoning.max_height
                }
                row.update(zoning_dict)
            
            return row
            
        except Exception as e:
            logger.error(f"Error extracting features for property {prop.id}: {e}")
            return None
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced features for ML model."""
        logger.info("🔧 Engineering advanced features...")
        
        df_engineered = df.copy()
        
        # Price per square foot (handle division by zero)
        df_engineered['price_per_sqft'] = np.where(
            df_engineered['square_feet'] > 0,
            df_engineered['list_price'] / df_engineered['square_feet'],
            np.nan
        )
        
        # Rent per square foot (handle division by zero)
        df_engineered['rent_per_sqft'] = np.where(
            df_engineered['square_feet'] > 0,
            df_engineered['annual_rent'] / df_engineered['square_feet'],
            np.nan
        )
        
        # Price to rent ratio (handle division by zero)
        df_engineered['price_to_rent_ratio'] = np.where(
            df_engineered['annual_rent'] > 0,
            df_engineered['list_price'] / df_engineered['annual_rent'],
            np.nan
        )
        
        # Property age
        current_year = datetime.now().year
        df_engineered['property_age'] = current_year - df_engineered['year_built']
        
        # Bedroom to bathroom ratio (handle division by zero)
        df_engineered['bed_bath_ratio'] = np.where(
            df_engineered['bathrooms'] > 0,
            df_engineered['bedrooms'] / df_engineered['bathrooms'],
            np.nan
        )
        
        # Income to expense ratio (handle missing columns)
        if 'total_income' in df_engineered.columns and 'total_expenses' in df_engineered.columns:
            df_engineered['income_expense_ratio'] = np.where(
                df_engineered['total_expenses'] > 0,
                df_engineered['total_income'] / df_engineered['total_expenses'],
                np.nan
            )
        else:
            df_engineered['income_expense_ratio'] = np.nan
        
        # Tax burden (taxes as % of income) - handle missing columns
        if 'annual_taxes' in df_engineered.columns and 'total_income' in df_engineered.columns:
            df_engineered['tax_burden'] = np.where(
                df_engineered['total_income'] > 0,
                df_engineered['annual_taxes'] / df_engineered['total_income'],
                np.nan
            )
        else:
            df_engineered['tax_burden'] = np.nan
        
        # Market indicators
        df_engineered['is_texas'] = df_engineered['state'].isin(['TX']).astype(int)
        df_engineered['is_sunbelt'] = df_engineered['state'].isin(['TX', 'AZ', 'CA', 'FL', 'NV']).astype(int)
        
        # Property type indicators
        df_engineered['is_single_family'] = df_engineered['property_type'].str.contains('Single Family', na=False).astype(int)
        df_engineered['is_multi_family'] = df_engineered['property_type'].str.contains('Multi', na=False).astype(int)
        df_engineered['is_commercial'] = df_engineered['property_type'].str.contains('Commercial', na=False).astype(int)
        
        # Condition indicators
        df_engineered['is_excellent'] = df_engineered['condition'].str.contains('Excellent', na=False).astype(int)
        df_engineered['is_good'] = df_engineered['condition'].str.contains('Good', na=False).astype(int)
        df_engineered['needs_work'] = df_engineered['condition'].str.contains('Needs Work', na=False).astype(int)
        
        # Feature count (handle missing features column)
        if 'features' in df_engineered.columns:
            df_engineered['feature_count'] = df_engineered['features'].apply(
                lambda x: len(json.loads(x)) if isinstance(x, str) else 0
            )
        else:
            df_engineered['feature_count'] = 0
        
        # Zoning indicators (handle missing zoning_code column)
        if 'zoning_code' in df_engineered.columns:
            df_engineered['is_residential_zoning'] = df_engineered['zoning_code'].str.contains('SF|MF|R', na=False).astype(int)
            df_engineered['is_commercial_zoning'] = df_engineered['zoning_code'].str.contains('C|I', na=False).astype(int)
        else:
            df_engineered['is_residential_zoning'] = 0
            df_engineered['is_commercial_zoning'] = 0
        
        # Development potential score (handle missing columns)
        development_components = []
        if 'max_density' in df_engineered.columns:
            development_components.append(df_engineered['max_density'] * 0.4)
        if 'max_height' in df_engineered.columns:
            development_components.append(df_engineered['max_height'] * 0.3)
        if 'feature_count' in df_engineered.columns:
            development_components.append(df_engineered['feature_count'] * 0.3)
        
        if development_components:
            df_engineered['development_potential'] = sum(development_components)
        else:
            df_engineered['development_potential'] = 0
        
        # Market efficiency score (handle missing columns)
        efficiency_components = []
        if 'cap_rate' in df_engineered.columns:
            efficiency_components.append(df_engineered['cap_rate'] * 0.4)
        if 'cash_on_cash_return' in df_engineered.columns:
            efficiency_components.append(df_engineered['cash_on_cash_return'] * 0.3)
        if 'risk_score' in df_engineered.columns:
            efficiency_components.append((100 - df_engineered['risk_score']) * 0.3)
        
        if efficiency_components:
            df_engineered['market_efficiency'] = sum(efficiency_components)
        else:
            df_engineered['market_efficiency'] = 0
        
        # Location score (simplified)
        location_scores = {
            'Austin': 85, 'Dallas': 80, 'Houston': 75, 'Phoenix': 70,
            'San Antonio': 75, 'Fort Worth': 70, 'Las Vegas': 65
        }
        df_engineered['location_score'] = df_engineered['city'].map(location_scores).fillna(50)
        
        # Investment grade score (handle missing columns)
        investment_components = []
        if 'cap_rate' in df_engineered.columns:
            investment_components.append(df_engineered['cap_rate'] * 0.25)
        if 'cash_on_cash_return' in df_engineered.columns:
            investment_components.append(df_engineered['cash_on_cash_return'] * 0.25)
        if 'location_score' in df_engineered.columns:
            investment_components.append(df_engineered['location_score'] * 0.25)
        if 'risk_score' in df_engineered.columns:
            investment_components.append((100 - df_engineered['risk_score']) * 0.25)
        
        if investment_components:
            df_engineered['investment_grade'] = sum(investment_components)
        else:
            df_engineered['investment_grade'] = 0
        
        logger.info(f"✅ Engineered {len(df_engineered.columns)} features")
        return df_engineered
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select the most important features for the model."""
        logger.info("🔧 Selecting features...")
        
        # Define feature columns (exclude non-numeric and target columns)
        exclude_columns = [
            'property_id', 'address', 'city', 'state', 'zip_code', 
            'property_type', 'condition', 'features', 'source',
            'zoning_code', 'land_use', 'assessment_year'
        ]
        
        self.feature_columns = [
            col for col in df.columns 
            if col not in exclude_columns and col != self.target_column
            and df[col].dtype in ['int64', 'float64']
        ]
        
        # Remove columns with too many missing values
        missing_threshold = 0.5
        for col in self.feature_columns[:]:
            missing_pct = df[col].isna().sum() / len(df)
            if missing_pct > missing_threshold:
                self.feature_columns.remove(col)
                logger.warning(f"Removed {col} due to {missing_pct:.1%} missing values")
        
        # Check if target column exists
        if self.target_column not in df.columns:
            logger.warning(f"Target column '{self.target_column}' not found. Available columns: {list(df.columns)}")
            # Try to find alternative target column
            if 'deal_score' in df.columns:
                self.target_column = 'deal_score'
            elif 'ml_score' in df.columns:
                self.target_column = 'ml_score'
            else:
                logger.error("No suitable target column found for ML training")
                return pd.DataFrame()
        
        # Fill missing values
        df_selected = df[self.feature_columns + [self.target_column]].copy()
        df_selected = df_selected.fillna(df_selected.median())
        
        logger.info(f"✅ Selected {len(self.feature_columns)} features")
        logger.info(f"Available columns: {list(df.columns)}")
        logger.info(f"Target column: {self.target_column}")
        return df_selected
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train multiple ML models and ensemble."""
        logger.info("🤖 Training advanced ML models...")
        
        if df.empty or len(df) < 10:
            logger.warning("Insufficient data for training")
            return {}
        
        # Prepare data
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['robust'] = scaler
        
        # Train individual models
        for model_name, config in self.model_configs.items():
            try:
                logger.info(f"Training {model_name}...")
                
                model = config['model'](**config['params'])
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                self.models[model_name] = model
                self.model_performance[model_name] = {
                    'mse': mse,
                    'r2': r2,
                    'mae': mae,
                    'rmse': np.sqrt(mse)
                }
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = dict(
                        zip(self.feature_columns, model.feature_importances_)
                    )
                
                logger.info(f"✅ {model_name} - R²: {r2:.3f}, RMSE: {np.sqrt(mse):.3f}")
                
            except Exception as e:
                logger.error(f"❌ Error training {model_name}: {e}")
        
        # Train ensemble model
        try:
            logger.info("Training ensemble model...")
            
            estimators = []
            for model_name, model in self.models.items():
                estimators.append((model_name, model))
            
            ensemble = VotingRegressor(estimators=estimators)
            ensemble.fit(X_train_scaled, y_train)
            
            # Evaluate ensemble
            y_pred_ensemble = ensemble.predict(X_test_scaled)
            mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
            r2_ensemble = r2_score(y_test, y_pred_ensemble)
            mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
            
            self.models['ensemble'] = ensemble
            self.model_performance['ensemble'] = {
                'mse': mse_ensemble,
                'r2': r2_ensemble,
                'mae': mae_ensemble,
                'rmse': np.sqrt(mse_ensemble)
            }
            
            logger.info(f"✅ Ensemble - R²: {r2_ensemble:.3f}, RMSE: {np.sqrt(mse_ensemble):.3f}")
            
        except Exception as e:
            logger.error(f"❌ Error training ensemble: {e}")
        
        return self.model_performance
    
    def predict_deal_score(self, property_id: int) -> Dict[str, Any]:
        """Predict deal score for a specific property."""
        try:
            # Get property data
            df = self.prepare_advanced_features()
            property_data = df[df['property_id'] == property_id]
            
            if property_data.empty:
                raise ValueError(f"Property {property_id} not found in dataset")
            
            # Prepare features
            X = property_data[self.feature_columns].fillna(property_data[self.feature_columns].median())
            
            # Scale features
            if 'robust' in self.scalers:
                X_scaled = self.scalers['robust'].transform(X)
            else:
                X_scaled = X
            
            # Make predictions with all models
            predictions = {}
            for model_name, model in self.models.items():
                try:
                    pred = model.predict(X_scaled)[0]
                    predictions[model_name] = max(0, min(100, pred))  # Clamp to 0-100
                except Exception as e:
                    logger.error(f"Error predicting with {model_name}: {e}")
                    predictions[model_name] = None
            
            # Use ensemble prediction as primary
            primary_prediction = predictions.get('ensemble', predictions.get('random_forest', 50))
            
            # Calculate confidence based on model agreement
            valid_predictions = [p for p in predictions.values() if p is not None]
            if len(valid_predictions) > 1:
                confidence = 100 - np.std(valid_predictions)
            else:
                confidence = 75  # Default confidence
            
            return {
                'property_id': property_id,
                'deal_score': primary_prediction,
                'confidence_score': max(0, min(100, confidence)),
                'model_predictions': predictions,
                'feature_importance': self.feature_importance.get('ensemble', {}),
                'prediction_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error predicting deal score for property {property_id}: {e}")
            return {
                'property_id': property_id,
                'deal_score': 50,
                'confidence_score': 0,
                'error': str(e)
            }
    
    def score_all_properties(self) -> List[Dict[str, Any]]:
        """Score all properties in the database."""
        logger.info("🤖 Scoring all properties...")
        
        try:
            df = self.prepare_advanced_features()
            
            if df.empty:
                logger.warning("No data available for scoring")
                return []
            
            results = []
            
            for property_id in df['property_id'].unique():
                try:
                    prediction = self.predict_deal_score(int(property_id))
                    results.append(prediction)
                except Exception as e:
                    logger.error(f"Error scoring property {property_id}: {e}")
                    continue
            
            logger.info(f"✅ Scored {len(results)} properties")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error scoring properties: {e}")
            return []
    
    def save_models(self, filepath: str = None):
        """Save trained models to disk."""
        if filepath is None:
            filepath = f"ml_models/saved_models/advanced_scorer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_importance': self.feature_importance,
                'model_performance': self.model_performance,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'trained_at': datetime.utcnow().isoformat()
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"✅ Models saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error saving models: {e}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk."""
        try:
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_importance = model_data['feature_importance']
            self.model_performance = model_data['model_performance']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            
            logger.info(f"✅ Models loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get summary of feature importance across all models."""
        if not self.feature_importance:
            return {}
        
        # Aggregate feature importance across models
        all_features = set()
        for model_importance in self.feature_importance.values():
            all_features.update(model_importance.keys())
        
        feature_summary = {}
        for feature in all_features:
            importance_values = []
            for model_importance in self.feature_importance.values():
                if feature in model_importance:
                    importance_values.append(model_importance[feature])
            
            if importance_values:
                feature_summary[feature] = {
                    'mean_importance': np.mean(importance_values),
                    'std_importance': np.std(importance_values),
                    'min_importance': np.min(importance_values),
                    'max_importance': np.max(importance_values)
                }
        
        # Sort by mean importance
        sorted_features = sorted(
            feature_summary.items(),
            key=lambda x: x[1]['mean_importance'],
            reverse=True
        )
        
        return dict(sorted_features)
    
    def generate_model_report(self) -> Dict[str, Any]:
        """Generate comprehensive model performance report."""
        report = {
            'model_performance': self.model_performance,
            'feature_importance_summary': self.get_feature_importance_summary(),
            'training_info': {
                'feature_count': len(self.feature_columns),
                'models_trained': len(self.models),
                'best_model': None,
                'best_r2': -1
            }
        }
        
        # Find best model
        for model_name, performance in self.model_performance.items():
            if performance['r2'] > report['training_info']['best_r2']:
                report['training_info']['best_r2'] = performance['r2']
                report['training_info']['best_model'] = model_name
        
        return report


if __name__ == "__main__":
    """Main execution for advanced ML model training and scoring."""
    from loguru import logger
    
    logger.info("🚀 Starting Advanced ML Model Training and Scoring")
    
    try:
        # Initialize advanced scorer
        advanced_scorer = AdvancedDealScorer()
        
        # Prepare features
        logger.info("📊 Preparing advanced features...")
        df = advanced_scorer.prepare_advanced_features()
        
        if df.empty:
            logger.warning("No data available for training. Please run data ingestion first.")
            exit(1)
        
        # Train models
        logger.info("🤖 Training advanced ML models...")
        performance = advanced_scorer.train_models(df)
        
        if performance:
            logger.info("✅ Model training completed successfully!")
            
            # Generate and display model report
            report = advanced_scorer.generate_model_report()
            
            logger.info("📈 Model Performance Summary:")
            for model_name, metrics in report['model_performance'].items():
                logger.info(f"  {model_name}: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}")
            
            # Score all properties
            logger.info("🎯 Scoring all properties with advanced models...")
            scores = advanced_scorer.score_all_properties()
            
            if scores:
                logger.info(f"✅ Successfully scored {len(scores)} properties")
                
                # Save models
                advanced_scorer.save_models()
                
                # Display top scores
                top_scores = sorted(scores, key=lambda x: x.get('deal_score', 0), reverse=True)[:5]
                logger.info("🏆 Top 5 Deal Scores:")
                for i, score in enumerate(top_scores, 1):
                    logger.info(f"  {i}. Property {score['property_id']}: {score['deal_score']:.1f} (Confidence: {score['confidence_score']:.1f})")
            else:
                logger.warning("No properties were scored")
        else:
            logger.error("❌ Model training failed")
            
    except Exception as e:
        logger.error(f"❌ Error in advanced ML processing: {e}")
        import traceback
        traceback.print_exc()
