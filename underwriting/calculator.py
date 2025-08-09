"""
Underwriting calculator for real estate investment analysis.
Performs financial calculations including NOI, Cap Rate, Cash-on-Cash return, and more.
"""

import math
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from loguru import logger

from database.connection import db_manager
from database.models import Property, UnderwritingData, TaxRecord


@dataclass
class UnderwritingAssumptions:
    """Default underwriting assumptions for calculations."""
    
    # Financing assumptions
    down_payment_percent: float = 0.25  # 25% down payment
    interest_rate: float = 0.065  # 6.5% interest rate
    loan_term_years: int = 30  # 30-year fixed
    
    # Operating expense assumptions
    property_management_fee_percent: float = 0.08  # 8% of gross rent
    maintenance_reserve_percent: float = 0.05  # 5% of gross rent
    insurance_rate_percent: float = 0.0035  # 0.35% of property value
    vacancy_rate_percent: float = 0.05  # 5% vacancy rate
    
    # Other assumptions
    closing_costs_percent: float = 0.03  # 3% of purchase price
    annual_appreciation_percent: float = 0.03  # 3% annual appreciation
    annual_rent_growth_percent: float = 0.02  # 2% annual rent growth


class UnderwritingCalculator:
    """Main underwriting calculator for real estate investment analysis."""
    
    def __init__(self, assumptions: Optional[UnderwritingAssumptions] = None):
        self.assumptions = assumptions or UnderwritingAssumptions()
    
    def calculate_all_metrics(self, property_id: int, custom_assumptions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate all underwriting metrics for a property."""
        try:
            # Get property data
            with db_manager.get_session() as session:
                property_obj = session.query(Property).filter(Property.id == property_id).first()
                if not property_obj:
                    raise ValueError(f"Property with ID {property_id} not found")
                
                # Get tax data
                tax_record = session.query(TaxRecord).filter(TaxRecord.property_id == property_id).first()
                
                # Merge custom assumptions with defaults
                merged_assumptions = self._merge_assumptions(custom_assumptions)
                
                # Calculate all metrics
                results = self._calculate_property_metrics(property_obj, tax_record, merged_assumptions)
                
                # Save results to database
                self._save_underwriting_data(property_id, results, session)
                
                return results
                
        except Exception as e:
            logger.error(f"Error calculating underwriting metrics: {e}")
            raise
    
    def _merge_assumptions(self, custom_assumptions: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge custom assumptions with default assumptions."""
        default_assumptions = {
            'down_payment_percent': self.assumptions.down_payment_percent,
            'interest_rate': self.assumptions.interest_rate,
            'loan_term_years': self.assumptions.loan_term_years,
            'property_management_fee_percent': self.assumptions.property_management_fee_percent,
            'maintenance_reserve_percent': self.assumptions.maintenance_reserve_percent,
            'insurance_rate_percent': self.assumptions.insurance_rate_percent,
            'vacancy_rate_percent': self.assumptions.vacancy_rate_percent,
            'closing_costs_percent': self.assumptions.closing_costs_percent,
            'annual_appreciation_percent': self.assumptions.annual_appreciation_percent,
            'annual_rent_growth_percent': self.assumptions.annual_rent_growth_percent,
        }
        
        if custom_assumptions:
            default_assumptions.update(custom_assumptions)
        
        return default_assumptions
    
    def _calculate_property_metrics(self, property_obj: Property, tax_record: Optional[TaxRecord], assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate all property investment metrics."""
        
        # Get base values
        purchase_price = property_obj.list_price or property_obj.estimated_value or 0
        monthly_rent = property_obj.monthly_rent or 0
        annual_rent = property_obj.annual_rent or (monthly_rent * 12)
        
        # Generate realistic rental estimates if missing
        if annual_rent <= 0 and purchase_price > 0:
            # Estimate rent based on property type and location
            rent_multiplier = self._estimate_rent_multiplier(property_obj)
            annual_rent = purchase_price * rent_multiplier
            monthly_rent = annual_rent / 12
        
        if purchase_price <= 0:
            raise ValueError("Invalid purchase price")
        
        if annual_rent <= 0:
            raise ValueError("Unable to estimate rental income")
        
        # Calculate financing
        down_payment = purchase_price * assumptions['down_payment_percent']
        loan_amount = purchase_price - down_payment
        monthly_payment = self._calculate_monthly_payment(loan_amount, assumptions['interest_rate'], assumptions['loan_term_years'])
        annual_payment = monthly_payment * 12
        
        # Calculate income
        gross_rental_income = annual_rent * (1 - assumptions['vacancy_rate_percent'])
        other_income = 0  # Could include parking, storage, etc.
        total_income = gross_rental_income + other_income
        
        # Calculate expenses
        property_management_fee = gross_rental_income * assumptions['property_management_fee_percent']
        maintenance_reserves = gross_rental_income * assumptions['maintenance_reserve_percent']
        property_taxes = tax_record.annual_taxes if tax_record else (purchase_price * 0.015)  # Estimate 1.5% if no tax data
        insurance = purchase_price * assumptions['insurance_rate_percent']
        utilities = 0  # Usually paid by tenant
        other_expenses = 0  # HOA, landscaping, etc.
        
        total_expenses = (
            property_management_fee +
            maintenance_reserves +
            property_taxes +
            insurance +
            utilities +
            other_expenses
        )
        
        # Calculate key metrics
        noi = total_income - total_expenses
        cap_rate = (noi / purchase_price) * 100 if purchase_price > 0 else 0
        
        # Calculate cash flow
        annual_cash_flow = noi - (annual_payment - (loan_amount * assumptions['interest_rate']))
        monthly_cash_flow = annual_cash_flow / 12
        
        # Calculate returns
        cash_on_cash_return = (annual_cash_flow / down_payment) * 100 if down_payment > 0 else 0
        
        # Calculate IRR (simplified)
        irr = self._calculate_simplified_irr(purchase_price, annual_cash_flow, assumptions)
        
        return {
            # Income analysis
            'gross_rental_income': gross_rental_income,
            'other_income': other_income,
            'total_income': total_income,
            
            # Expense analysis
            'property_management_fee': property_management_fee,
            'maintenance_reserves': maintenance_reserves,
            'property_taxes': property_taxes,
            'insurance': insurance,
            'utilities': utilities,
            'other_expenses': other_expenses,
            'total_expenses': total_expenses,
            
            # Key metrics
            'noi': noi,
            'cap_rate': cap_rate,
            'cash_on_cash_return': cash_on_cash_return,
            'internal_rate_of_return': irr,
            
            # Purchase assumptions
            'purchase_price': purchase_price,
            'down_payment': down_payment,
            'loan_amount': loan_amount,
            'interest_rate': assumptions['interest_rate'],
            'loan_term': assumptions['loan_term_years'],
            
            # Monthly payments
            'principal_interest': monthly_payment,
            'property_tax_payment': property_taxes / 12,
            'insurance_payment': insurance / 12,
            'total_monthly_payment': monthly_payment + (property_taxes / 12) + (insurance / 12),
            
            # Cash flow
            'monthly_cash_flow': monthly_cash_flow,
            'annual_cash_flow': annual_cash_flow,
        }
    
    def _estimate_rent_multiplier(self, property_obj: Property) -> float:
        """Estimate annual rent as a percentage of property value based on property characteristics."""
        
        # Base rent multiplier (annual rent / property value)
        base_multiplier = 0.06  # 6% annual rent to value ratio
        
        # Adjust based on property type
        property_type = property_obj.property_type or ""
        if "Single Family" in property_type:
            multiplier = 0.055  # 5.5% for single family
        elif "Multi" in property_type:
            multiplier = 0.065  # 6.5% for multi-family
        elif "Commercial" in property_type:
            multiplier = 0.075  # 7.5% for commercial
        else:
            multiplier = base_multiplier
        
        # Adjust based on location (market-specific)
        state = property_obj.state or ""
        if state in ["TX", "AZ", "FL"]:  # Sunbelt markets
            multiplier *= 1.1  # 10% higher rent ratios
        elif state in ["CA", "NY"]:  # High-cost markets
            multiplier *= 0.9  # 10% lower rent ratios
        
        # Adjust based on property age
        if property_obj.year_built:
            age = 2024 - property_obj.year_built
            if age < 10:
                multiplier *= 1.05  # 5% higher for newer properties
            elif age > 30:
                multiplier *= 0.95  # 5% lower for older properties
        
        return multiplier
    
    def _calculate_monthly_payment(self, loan_amount: float, interest_rate: float, loan_term_years: int) -> float:
        """Calculate monthly mortgage payment."""
        if loan_amount <= 0 or interest_rate <= 0 or loan_term_years <= 0:
            return 0
        
        monthly_rate = interest_rate / 12 / 100
        num_payments = loan_term_years * 12
        
        if monthly_rate == 0:
            return loan_amount / num_payments
        
        payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)
        return payment
    
    def _calculate_simplified_irr(self, purchase_price: float, annual_cash_flow: float, assumptions: Dict[str, Any]) -> float:
        """Calculate simplified IRR (Internal Rate of Return)."""
        try:
            # This is a simplified IRR calculation
            # In practice, you'd want to use a more sophisticated method
            appreciation = purchase_price * assumptions['annual_appreciation_percent']
            total_return = annual_cash_flow + appreciation
            irr = (total_return / purchase_price) * 100
            return irr
        except Exception:
            return 0
    
    def _save_underwriting_data(self, property_id: int, results: Dict[str, Any], session):
        """Save underwriting results to database."""
        try:
            # Check if underwriting data already exists
            existing_data = session.query(UnderwritingData).filter(
                UnderwritingData.property_id == property_id
            ).first()
            
            if existing_data:
                # Update existing record
                for key, value in results.items():
                    if hasattr(existing_data, key):
                        setattr(existing_data, key, value)
            else:
                # Create new record
                underwriting_data = UnderwritingData(
                    property_id=property_id,
                    **results
                )
                session.add(underwriting_data)
            
            session.commit()
            logger.info(f"✅ Saved underwriting data for property {property_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save underwriting data: {e}")
            session.rollback()
            raise
    
    def calculate_portfolio_metrics(self, property_ids: List[int]) -> Dict[str, Any]:
        """Calculate portfolio-level metrics for multiple properties."""
        portfolio_results = {
            'total_properties': len(property_ids),
            'total_investment': 0,
            'total_noi': 0,
            'total_cash_flow': 0,
            'weighted_cap_rate': 0,
            'average_cash_on_cash': 0,
            'properties': []
        }
        
        total_value = 0
        
        for property_id in property_ids:
            try:
                property_metrics = self.calculate_all_metrics(property_id)
                portfolio_results['properties'].append({
                    'property_id': property_id,
                    'metrics': property_metrics
                })
                
                # Aggregate metrics
                purchase_price = property_metrics['purchase_price']
                noi = property_metrics['noi']
                cash_flow = property_metrics['annual_cash_flow']
                cap_rate = property_metrics['cap_rate']
                coc_return = property_metrics['cash_on_cash_return']
                
                portfolio_results['total_investment'] += purchase_price
                portfolio_results['total_noi'] += noi
                portfolio_results['total_cash_flow'] += cash_flow
                total_value += purchase_price
                
            except Exception as e:
                logger.error(f"Error calculating metrics for property {property_id}: {e}")
                continue
        
        # Calculate weighted averages
        if total_value > 0:
            portfolio_results['weighted_cap_rate'] = (portfolio_results['total_noi'] / total_value) * 100
        
        if portfolio_results['total_properties'] > 0:
            portfolio_results['average_cash_on_cash'] = sum(
                p['metrics']['cash_on_cash_return'] for p in portfolio_results['properties']
            ) / portfolio_results['total_properties']
        
        return portfolio_results
    
    def generate_investment_report(self, property_id: int) -> Dict[str, Any]:
        """Generate a comprehensive investment report for a property."""
        try:
            # Calculate metrics
            metrics = self.calculate_all_metrics(property_id)
            
            # Get property details
            with db_manager.get_session() as session:
                property_obj = session.query(Property).filter(Property.id == property_id).first()
                
                if not property_obj:
                    raise ValueError(f"Property with ID {property_id} not found")
                
                # Generate report
                report = {
                    'property_info': {
                        'id': property_obj.id,
                        'address': property_obj.address,
                        'city': property_obj.city,
                        'state': property_obj.state,
                        'zip_code': property_obj.zip_code,
                        'property_type': property_obj.property_type,
                        'bedrooms': property_obj.bedrooms,
                        'bathrooms': property_obj.bathrooms,
                        'square_feet': property_obj.square_feet,
                        'year_built': property_obj.year_built,
                    },
                    'financial_metrics': metrics,
                    'investment_grade': self._calculate_investment_grade(metrics),
                    'risk_assessment': self._assess_investment_risk(metrics),
                    'recommendations': self._generate_recommendations(metrics),
                }
                
                return report
                
        except Exception as e:
            logger.error(f"Error generating investment report: {e}")
            raise
    
    def _calculate_investment_grade(self, metrics: Dict[str, Any]) -> str:
        """Calculate investment grade based on metrics."""
        cap_rate = metrics.get('cap_rate', 0)
        cash_on_cash = metrics.get('cash_on_cash_return', 0)
        
        if cap_rate >= 8 and cash_on_cash >= 10:
            return "A"
        elif cap_rate >= 6 and cash_on_cash >= 7:
            return "B"
        elif cap_rate >= 4 and cash_on_cash >= 5:
            return "C"
        else:
            return "D"
    
    def _assess_investment_risk(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess investment risk based on metrics."""
        risk_factors = []
        risk_score = 0
        
        # Cap rate risk
        cap_rate = metrics.get('cap_rate', 0)
        if cap_rate < 4:
            risk_factors.append("Low cap rate indicates potential overvaluation")
            risk_score += 3
        elif cap_rate > 12:
            risk_factors.append("High cap rate may indicate high-risk area")
            risk_score += 2
        
        # Cash flow risk
        monthly_cash_flow = metrics.get('monthly_cash_flow', 0)
        if monthly_cash_flow < 0:
            risk_factors.append("Negative cash flow")
            risk_score += 5
        elif monthly_cash_flow < 200:
            risk_factors.append("Low cash flow margin")
            risk_score += 2
        
        # Leverage risk
        down_payment_percent = (metrics.get('down_payment', 0) / metrics.get('purchase_price', 1)) * 100
        if down_payment_percent < 20:
            risk_factors.append("High leverage (low down payment)")
            risk_score += 2
        
        return {
            'risk_score': min(risk_score, 10),  # Scale 0-10
            'risk_level': self._get_risk_level(risk_score),
            'risk_factors': risk_factors,
        }
    
    def _get_risk_level(self, risk_score: int) -> str:
        """Get risk level based on risk score."""
        if risk_score <= 2:
            return "Low"
        elif risk_score <= 5:
            return "Medium"
        elif risk_score <= 8:
            return "High"
        else:
            return "Very High"
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate investment recommendations based on metrics."""
        recommendations = []
        
        cap_rate = metrics.get('cap_rate', 0)
        cash_on_cash = metrics.get('cash_on_cash_return', 0)
        monthly_cash_flow = metrics.get('monthly_cash_flow', 0)
        
        if cap_rate < 6:
            recommendations.append("Consider negotiating a lower purchase price to improve cap rate")
        
        if cash_on_cash < 8:
            recommendations.append("Cash-on-cash return is below recommended threshold")
        
        if monthly_cash_flow < 0:
            recommendations.append("Property has negative cash flow - consider as appreciation play only")
        
        if monthly_cash_flow > 500:
            recommendations.append("Strong positive cash flow - good investment candidate")
        
        return recommendations
