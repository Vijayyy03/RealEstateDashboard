"""
Machine Learning model for scoring and ranking real estate investment deals.
Uses scikit-learn to create a comprehensive deal scoring system.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
from loguru import logger

from config.settings import settings
from database.connection import db_manager
from database.models import Property, UnderwritingData, MLScore


class DealScorer:
    """Machine Learning model for scoring real estate investment deals."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.label_encoders = {}
        self.feature_columns = settings.ml.feature_columns
        self.model_path = os.path.join(settings.ml.model_dir, f"{settings.ml.scoring_model_name}.pkl")
        self.scaler_path = os.path.join(settings.ml.model_dir, f"{settings.ml.scoring_model_name}_scaler.pkl")
        
        # Create model directory if it doesn't exist
        os.makedirs(settings.ml.model_dir, exist_ok=True)
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from database."""
        try:
            with db_manager.get_session() as session:
                # Query properties with underwriting data
                query = session.query(
                    Property, UnderwritingData
                ).join(
                    UnderwritingData, Property.id == UnderwritingData.property_id
                ).filter(
                    Property.is_active == True
                )
                
                # Convert to DataFrame
                data = []
                for property_obj, underwriting_obj in query.all():
                    row = self._extract_features(property_obj, underwriting_obj)
                    if row:
                        data.append(row)
                
                if not data:
                    raise ValueError("No training data available")
                
                df = pd.DataFrame(data)
                logger.info(f"Prepared {len(df)} training samples")
                
                # Prepare features and target
                X = self._prepare_features(df)
                y = self._create_target_variable(df)
                
                return X, y
                
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
    
    def _extract_features(self, property_obj: Property, underwriting_obj: UnderwritingData) -> Optional[Dict[str, Any]]:
        """Extract features from property and underwriting data."""
        try:
            # Basic property features
            features = {
                'price': property_obj.list_price or property_obj.estimated_value or 0,
                'sqft': property_obj.square_feet or 0,
                'bedrooms': property_obj.bedrooms or 0,
                'bathrooms': property_obj.bathrooms or 0,
                'year_built': property_obj.year_built or 0,
                'lot_size': property_obj.lot_size or 0,
            }
            
            # Financial features
            if underwriting_obj:
                features.update({
                    'cap_rate': underwriting_obj.cap_rate or 0,
                    'noi': underwriting_obj.noi or 0,
                    'cash_on_cash': underwriting_obj.cash_on_cash_return or 0,
                    'monthly_cash_flow': underwriting_obj.monthly_cash_flow or 0,
                    'annual_cash_flow': underwriting_obj.annual_cash_flow or 0,
                    'total_income': underwriting_obj.total_income or 0,
                    'total_expenses': underwriting_obj.total_expenses or 0,
                    'purchase_price': underwriting_obj.purchase_price or 0,
                    'down_payment': underwriting_obj.down_payment or 0,
                    'loan_amount': underwriting_obj.loan_amount or 0,
                })
            
            # Derived features
            features.update(self._calculate_derived_features(features))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _calculate_derived_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived features from base features."""
        derived = {}
        
        # Price per square foot
        if features.get('sqft', 0) > 0:
            derived['price_per_sqft'] = features.get('price', 0) / features.get('sqft', 1)
        else:
            derived['price_per_sqft'] = 0
        
        # Rent per square foot (if available)
        if features.get('sqft', 0) > 0 and features.get('total_income', 0) > 0:
            derived['rent_per_sqft'] = features.get('total_income', 0) / features.get('sqft', 1)
        else:
            derived['rent_per_sqft'] = 0
        
        # Loan-to-value ratio
        if features.get('purchase_price', 0) > 0:
            derived['ltv_ratio'] = features.get('loan_amount', 0) / features.get('purchase_price', 1)
        else:
            derived['ltv_ratio'] = 0
        
        # Expense ratio
        if features.get('total_income', 0) > 0:
            derived['expense_ratio'] = features.get('total_expenses', 0) / features.get('total_income', 1)
        else:
            derived['expense_ratio'] = 0
        
        # Property age
        current_year = datetime.now().year
        derived['property_age'] = current_year - features.get('year_built', current_year)
        
        # Bedroom to bathroom ratio
        if features.get('bathrooms', 0) > 0:
            derived['bed_bath_ratio'] = features.get('bedrooms', 0) / features.get('bathrooms', 1)
        else:
            derived['bed_bath_ratio'] = 0
        
        return derived
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model training."""
        # Select numeric features
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target variable if present
        if 'deal_score' in numeric_features:
            numeric_features.remove('deal_score')
        
        # Handle missing values
        df_clean = df[numeric_features].fillna(0)
        
        # Remove infinite values
        df_clean = df_clean.replace([np.inf, -np.inf], 0)
        
        # Feature selection
        if len(numeric_features) > 10:
            self.feature_selector = SelectKBest(score_func=f_regression, k=10)
            target = df.get('deal_score', pd.Series([0] * len(df)))
            df_clean = pd.DataFrame(
                self.feature_selector.fit_transform(df_clean, target),
                columns=numeric_features[:10]
            )
        
        # Scale features
        df_scaled = pd.DataFrame(
            self.scaler.fit_transform(df_clean),
            columns=df_clean.columns
        )
        
        return df_scaled
    
    def _create_target_variable(self, df: pd.DataFrame) -> pd.Series:
        """Create target variable for training."""
        # Create a composite score based on multiple factors
        scores = []
        
        for _, row in df.iterrows():
            score = self._calculate_composite_score(row)
            scores.append(score)
        
        return pd.Series(scores, name='deal_score')
    
    def _calculate_composite_score(self, row: pd.Series) -> float:
        """Calculate composite deal score from 0-100."""
        score = 0
        
        # Cap rate score (0-25 points)
        cap_rate = row.get('cap_rate', 0)
        if 6 <= cap_rate <= 10:
            score += 25
        elif 4 <= cap_rate < 6 or 10 < cap_rate <= 12:
            score += 15
        elif 2 <= cap_rate < 4 or 12 < cap_rate <= 15:
            score += 10
        else:
            score += 5
        
        # Cash-on-cash return score (0-25 points)
        coc_return = row.get('cash_on_cash', 0)
        if coc_return >= 10:
            score += 25
        elif 7 <= coc_return < 10:
            score += 20
        elif 5 <= coc_return < 7:
            score += 15
        elif 3 <= coc_return < 5:
            score += 10
        else:
            score += 5
        
        # Cash flow score (0-20 points)
        monthly_cf = row.get('monthly_cash_flow', 0)
        if monthly_cf >= 500:
            score += 20
        elif 200 <= monthly_cf < 500:
            score += 15
        elif 0 <= monthly_cf < 200:
            score += 10
        else:
            score += 0
        
        # Property condition score (0-15 points)
        year_built = row.get('year_built', 0)
        if year_built >= 2000:
            score += 15
        elif 1980 <= year_built < 2000:
            score += 12
        elif 1960 <= year_built < 1980:
            score += 8
        else:
            score += 5
        
        # Location score (0-15 points) - simplified
        # In a real implementation, this would use market data
        score += 10  # Default location score
        
        return min(score, 100)  # Cap at 100
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train the deal scoring model."""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=settings.ml.test_size, random_state=settings.ml.random_state
            )
            
            # Initialize model (using Random Forest for better interpretability)
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=settings.ml.random_state,
                n_jobs=-1
            )
            
            # Train model
            logger.info("Training deal scoring model...")
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X, y, cv=5)
            
            # Predictions
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Feature importance
            feature_importance = dict(zip(X.columns, self.model.feature_importances_))
            
            # Save model
            self.save_model()
            
            results = {
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'mse': mse,
                'mae': mae,
                'feature_importance': feature_importance,
                'model_version': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            logger.info(f"✅ Model training completed. Test R²: {test_score:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            raise
    
    def predict_deal_score(self, property_id: int) -> Dict[str, Any]:
        """Predict deal score for a specific property."""
        try:
            # Get property data
            with db_manager.get_session() as session:
                property_obj = session.query(Property).filter(Property.id == property_id).first()
                underwriting_obj = session.query(UnderwritingData).filter(
                    UnderwritingData.property_id == property_id
                ).first()
                
                if not property_obj or not underwriting_obj:
                    raise ValueError(f"Property {property_id} not found or missing underwriting data")
                
                # Extract features
                features = self._extract_features(property_obj, underwriting_obj)
                if not features:
                    raise ValueError("Could not extract features for property")
                
                # Prepare features
                df = pd.DataFrame([features])
                X = self._prepare_features(df)
                
                # Make prediction
                if self.model is None:
                    self.load_model()
                
                predicted_score = self.model.predict(X)[0]
                
                # Get feature importance for this prediction
                feature_importance = self._get_prediction_importance(X)
                
                # Calculate confidence (simplified)
                confidence = self._calculate_prediction_confidence(predicted_score, features)
                
                # Save prediction
                self._save_prediction(property_id, predicted_score, confidence, feature_importance, session)
                
                return {
                    'property_id': property_id,
                    'deal_score': predicted_score,
                    'confidence': confidence,
                    'feature_importance': feature_importance,
                    'prediction_date': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error predicting deal score: {e}")
            raise
    
    def _get_prediction_importance(self, X: pd.DataFrame) -> Dict[str, float]:
        """Get feature importance for a specific prediction."""
        if self.model is None:
            return {}
        
        importance = dict(zip(X.columns, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def _calculate_prediction_confidence(self, predicted_score: float, features: Dict[str, Any]) -> float:
        """Calculate confidence in prediction (0-1 scale)."""
        # Simplified confidence calculation
        # In practice, you might use model uncertainty or ensemble variance
        
        confidence = 0.7  # Base confidence
        
        # Adjust based on data quality
        missing_features = sum(1 for v in features.values() if v is None or v == 0)
        total_features = len(features)
        
        if total_features > 0:
            data_quality = 1 - (missing_features / total_features)
            confidence *= data_quality
        
        # Adjust based on score range
        if 20 <= predicted_score <= 80:
            confidence *= 1.0
        elif 10 <= predicted_score <= 90:
            confidence *= 0.9
        else:
            confidence *= 0.8
        
        return min(confidence, 1.0)
    
    def _save_prediction(self, property_id: int, deal_score: float, confidence: float, 
                        feature_importance: Dict[str, float], session):
        """Save prediction to database."""
        try:
            # Convert feature importance dict to JSON string for SQLite
            import json
            feature_importance_json = json.dumps(feature_importance)
            
            # Check if prediction already exists
            existing_prediction = session.query(MLScore).filter(
                MLScore.property_id == property_id,
                MLScore.model_name == settings.ml.scoring_model_name
            ).first()
            
            if existing_prediction:
                # Update existing prediction
                existing_prediction.deal_score = deal_score
                existing_prediction.confidence_score = confidence
                existing_prediction.feature_importance = feature_importance_json
                existing_prediction.updated_at = datetime.utcnow()
            else:
                # Create new prediction
                prediction = MLScore(
                    property_id=property_id,
                    model_name=settings.ml.scoring_model_name,
                    model_version=datetime.now().strftime("%Y%m%d"),
                    deal_score=deal_score,
                    confidence_score=confidence,
                    feature_importance=feature_importance_json
                )
                session.add(prediction)
            
            session.commit()
            logger.info(f"✅ Saved deal score prediction for property {property_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save prediction: {e}")
            session.rollback()
            raise
    
    def save_model(self):
        """Save trained model to disk."""
        try:
            # Save model
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save scaler
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            logger.info(f"✅ Model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            raise
    
    def load_model(self):
        """Load trained model from disk."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                logger.info("✅ Model loaded successfully")
            else:
                raise FileNotFoundError("Model file not found")
                
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def score_all_properties(self) -> List[Dict[str, Any]]:
        """Score all properties in the database."""
        try:
            with db_manager.get_session() as session:
                # Get all properties with underwriting data
                properties = session.query(Property).join(UnderwritingData).filter(
                    Property.is_active == True
                ).all()
                
                results = []
                for property_obj in properties:
                    try:
                        prediction = self.predict_deal_score(property_obj.id)
                        results.append(prediction)
                    except Exception as e:
                        logger.error(f"Error scoring property {property_obj.id}: {e}")
                        continue
                
                logger.info(f"✅ Scored {len(results)} properties")
                return results
                
        except Exception as e:
            logger.error(f"Error scoring all properties: {e}")
            raise
