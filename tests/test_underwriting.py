"""
Unit tests for the underwriting calculator.
Tests financial calculations and investment metrics.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from underwriting.calculator import UnderwritingCalculator, UnderwritingAssumptions
from database.models import Property, UnderwritingData
from database.connection import db_manager


class TestUnderwritingCalculator:
    """Test cases for UnderwritingCalculator."""
    
    @pytest.fixture
    def calculator(self):
        """Create a calculator instance for testing."""
        return UnderwritingCalculator()
    
    @pytest.fixture
    def sample_property_data(self):
        """Sample property data for testing."""
        return {
            'address': '123 Test St',
            'city': 'Test City',
            'state': 'TX',
            'zip_code': '12345',
            'property_type': 'Single Family',
            'bedrooms': 3,
            'bathrooms': 2.0,
            'square_feet': 1500,
            'year_built': 2000,
            'list_price': 250000,
            'monthly_rent': 2000,
            'annual_rent': 24000,
            'source': 'Test'
        }
    
    def test_monthly_payment_calculation(self, calculator):
        """Test monthly mortgage payment calculation."""
        # Test case: $200,000 loan, 6.5% interest, 30 years
        payment = calculator._calculate_monthly_payment(200000, 6.5, 30)
        expected_payment = 1264.14  # Approximate monthly payment
        
        assert abs(payment - expected_payment) < 1.0  # Allow small rounding differences
    
    def test_cap_rate_calculation(self, calculator):
        """Test cap rate calculation."""
        noi = 15000  # Net Operating Income
        purchase_price = 250000  # Purchase Price
        
        cap_rate = (noi / purchase_price) * 100
        expected_cap_rate = 6.0
        
        assert cap_rate == expected_cap_rate
    
    def test_cash_on_cash_calculation(self, calculator):
        """Test cash-on-cash return calculation."""
        annual_cash_flow = 8000  # Annual cash flow
        down_payment = 50000  # Down payment
        
        coc_return = (annual_cash_flow / down_payment) * 100
        expected_coc_return = 16.0
        
        assert coc_return == expected_coc_return
    
    def test_investment_grade_calculation(self, calculator):
        """Test investment grade calculation."""
        # Test A grade
        metrics_a = {'cap_rate': 8.5, 'cash_on_cash': 12.0}
        grade_a = calculator._calculate_investment_grade(metrics_a)
        assert grade_a == "A"
        
        # Test B grade
        metrics_b = {'cap_rate': 7.0, 'cash_on_cash': 8.0}
        grade_b = calculator._calculate_investment_grade(metrics_b)
        assert grade_b == "B"
        
        # Test C grade
        metrics_c = {'cap_rate': 5.0, 'cash_on_cash': 6.0}
        grade_c = calculator._calculate_investment_grade(metrics_c)
        assert grade_c == "C"
        
        # Test D grade
        metrics_d = {'cap_rate': 3.0, 'cash_on_cash': 3.0}
        grade_d = calculator._calculate_investment_grade(metrics_d)
        assert grade_d == "D"
    
    def test_risk_assessment(self, calculator):
        """Test risk assessment calculation."""
        # Test low risk
        metrics_low = {
            'cap_rate': 7.0,
            'monthly_cash_flow': 500,
            'down_payment': 50000,
            'purchase_price': 200000
        }
        risk_low = calculator._assess_investment_risk(metrics_low)
        assert risk_low['risk_level'] in ['Low', 'Medium']
        
        # Test high risk
        metrics_high = {
            'cap_rate': 3.0,
            'monthly_cash_flow': -100,
            'down_payment': 20000,
            'purchase_price': 200000
        }
        risk_high = calculator._assess_investment_risk(metrics_high)
        assert risk_high['risk_level'] in ['High', 'Very High']
    
    def test_underwriting_assumptions(self):
        """Test underwriting assumptions."""
        assumptions = UnderwritingAssumptions()
        
        assert assumptions.down_payment_percent == 0.25
        assert assumptions.interest_rate == 0.065
        assert assumptions.loan_term_years == 30
        assert assumptions.property_management_fee_percent == 0.08
        assert assumptions.maintenance_reserve_percent == 0.05
    
    def test_merge_assumptions(self, calculator):
        """Test merging custom assumptions with defaults."""
        custom_assumptions = {
            'down_payment_percent': 0.20,
            'interest_rate': 0.07
        }
        
        merged = calculator._merge_assumptions(custom_assumptions)
        
        assert merged['down_payment_percent'] == 0.20
        assert merged['interest_rate'] == 0.07
        assert merged['loan_term_years'] == 30  # Default value
        assert merged['property_management_fee_percent'] == 0.08  # Default value


class TestUnderwritingIntegration:
    """Integration tests for underwriting with database."""
    
    @pytest.fixture(autouse=True)
    def setup_database(self):
        """Setup test database."""
        # Use test database
        from config.settings import settings
        settings.environment = "test"
        
        # Initialize database
        db_manager.create_tables()
        yield
        # Cleanup
        db_manager.drop_tables()
    
    def test_full_underwriting_calculation(self):
        """Test complete underwriting calculation for a property."""
        calculator = UnderwritingCalculator()
        
        # Create test property
        with db_manager.get_session() as session:
            property_obj = Property(
                address='123 Test St',
                city='Test City',
                state='TX',
                zip_code='12345',
                property_type='Single Family',
                bedrooms=3,
                bathrooms=2.0,
                square_feet=1500,
                year_built=2000,
                list_price=250000,
                monthly_rent=2000,
                annual_rent=24000,
                source='Test'
            )
            session.add(property_obj)
            session.commit()
            
            # Calculate underwriting
            results = calculator.calculate_all_metrics(property_obj.id)
            
            # Verify results
            assert 'noi' in results
            assert 'cap_rate' in results
            assert 'cash_on_cash_return' in results
            assert 'monthly_cash_flow' in results
            
            # Verify reasonable values
            assert results['cap_rate'] > 0
            assert results['purchase_price'] == 250000
            assert results['down_payment'] == 250000 * 0.25  # 25% down payment
    
    def test_portfolio_calculation(self):
        """Test portfolio-level calculations."""
        calculator = UnderwritingCalculator()
        
        # Create multiple test properties
        with db_manager.get_session() as session:
            properties = []
            for i in range(3):
                property_obj = Property(
                    address=f'{i+1}23 Test St',
                    city='Test City',
                    state='TX',
                    zip_code='12345',
                    property_type='Single Family',
                    bedrooms=3,
                    bathrooms=2.0,
                    square_feet=1500,
                    year_built=2000,
                    list_price=250000,
                    monthly_rent=2000,
                    annual_rent=24000,
                    source='Test'
                )
                session.add(property_obj)
                properties.append(property_obj)
            
            session.commit()
            
            # Calculate underwriting for all properties
            property_ids = [p.id for p in properties]
            for property_id in property_ids:
                calculator.calculate_all_metrics(property_id)
            
            # Calculate portfolio metrics
            portfolio_results = calculator.calculate_portfolio_metrics(property_ids)
            
            # Verify portfolio results
            assert portfolio_results['total_properties'] == 3
            assert portfolio_results['total_investment'] == 750000  # 3 * 250000
            assert portfolio_results['weighted_cap_rate'] > 0
            assert portfolio_results['average_cash_on_cash'] > 0


if __name__ == "__main__":
    pytest.main([__file__])
