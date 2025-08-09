"""
Streamlit Dashboard for Real Estate Investment Decision System.
Provides interactive interface for viewing and analyzing investment opportunities.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from database.connection import db_manager
from database.models import Property, UnderwritingData, MLScore, MarketData
from underwriting.calculator import UnderwritingCalculator
from ml_models.deal_scorer import DealScorer


# Page configuration
st.set_page_config(
    page_title=settings.dashboard.title,
    page_icon=settings.dashboard.page_icon,
    layout=settings.dashboard.layout,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .deal-score-high {
        color: #28a745;
        font-weight: bold;
    }
    .deal-score-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .deal-score-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_property_data():
    """Load property data from database."""
    try:
        with db_manager.get_session() as session:
            # Query properties with underwriting and ML scores
            query = session.query(
                Property, UnderwritingData, MLScore
            ).outerjoin(
                UnderwritingData, Property.id == UnderwritingData.property_id
            ).outerjoin(
                MLScore, Property.id == MLScore.property_id
            ).filter(
                Property.is_active == True
            )
            
            data = []
            for property_obj, underwriting_obj, ml_score_obj in query.all():
                row = {
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
                    'list_price': property_obj.list_price,
                    'monthly_rent': property_obj.monthly_rent,
                    'annual_rent': property_obj.annual_rent,
                }
                
                # Add underwriting data
                if underwriting_obj:
                    row.update({
                        'cap_rate': underwriting_obj.cap_rate,
                        'noi': underwriting_obj.noi,
                        'cash_on_cash': underwriting_obj.cash_on_cash_return,
                        'monthly_cash_flow': underwriting_obj.monthly_cash_flow,
                        'annual_cash_flow': underwriting_obj.annual_cash_flow,
                        'total_income': underwriting_obj.total_income,
                        'total_expenses': underwriting_obj.total_expenses,
                    })
                
                # Add ML scores
                if ml_score_obj:
                    row.update({
                        'deal_score': ml_score_obj.deal_score,
                        'risk_score': ml_score_obj.risk_score,
                        'confidence': ml_score_obj.confidence_score,
                    })
                
                data.append(row)
            
            return pd.DataFrame(data)
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">🏠 Real Estate Investment Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_property_data()
    
    if df.empty:
        st.warning("No property data available. Please run data ingestion first.")
        return
    
    # Sidebar filters
    st.sidebar.header("📊 Filters")
    
    # Location filters
    st.sidebar.subheader("📍 Location")
    cities = ['All'] + sorted(df['city'].dropna().unique().tolist())
    selected_city = st.sidebar.selectbox("City", cities)
    
    states = ['All'] + sorted(df['state'].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox("State", states)
    
    # Property type filter
    st.sidebar.subheader("🏘️ Property Type")
    property_types = ['All'] + sorted(df['property_type'].dropna().unique().tolist())
    selected_type = st.sidebar.selectbox("Property Type", property_types)
    
    # Price range filter
    st.sidebar.subheader("💰 Price Range")
    min_price = float(df['list_price'].min()) if not df['list_price'].isna().all() else 0
    max_price = float(df['list_price'].max()) if not df['list_price'].isna().all() else 1000000
    
    # Ensure min_price is less than max_price
    if min_price >= max_price:
        min_price = max_price - 10000 if max_price > 10000 else 0
        max_price = max_price if max_price > 10000 else 10000
    
    price_range = st.sidebar.slider(
        "Price Range (₹)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=100000.0  # Increased step size for Indian prices
    )
    
    # Deal score filter
    st.sidebar.subheader("🎯 Deal Score")
    min_score = float(df['deal_score'].min()) if not df['deal_score'].isna().all() else 0
    max_score = float(df['deal_score'].max()) if not df['deal_score'].isna().all() else 100
    
    # Ensure min_score is less than max_score
    if min_score >= max_score:
        min_score = max_score - 1 if max_score > 1 else 0
        max_score = max_score if max_score > 1 else 1
    
    score_range = st.sidebar.slider(
        "Deal Score",
        min_value=min_score,
        max_value=max_score,
        value=(min_score, max_score),
        step=1.0
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_city != 'All':
        filtered_df = filtered_df[filtered_df['city'] == selected_city]
    
    if selected_state != 'All':
        filtered_df = filtered_df[filtered_df['state'] == selected_state]
    
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['property_type'] == selected_type]
    
    filtered_df = filtered_df[
        (filtered_df['list_price'] >= price_range[0]) &
        (filtered_df['list_price'] <= price_range[1])
    ]
    
    filtered_df = filtered_df[
        (filtered_df['deal_score'] >= score_range[0]) &
        (filtered_df['deal_score'] <= score_range[1])
    ]
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties", len(filtered_df))
    
    with col2:
        avg_price = filtered_df['list_price'].mean()
        st.metric("Avg Price", f"₹{avg_price:,.0f}" if not pd.isna(avg_price) else "N/A")
    
    with col3:
        avg_cap_rate = filtered_df['cap_rate'].mean()
        st.metric("Avg Cap Rate", f"{avg_cap_rate:.1f}%" if not pd.isna(avg_cap_rate) else "N/A")
    
    with col4:
        avg_deal_score = filtered_df['deal_score'].mean()
        st.metric("Avg Deal Score", f"{avg_deal_score:.1f}" if not pd.isna(avg_deal_score) else "N/A")
    
    # Charts
    st.header("📈 Investment Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Deal Rankings", "Market Analysis", "Financial Metrics", "Property Details", 
        "Portfolio Analysis", "AI Recommendations", "Comparison Tool"
    ])
    
    with tab1:
        display_deal_rankings(filtered_df)
    
    with tab2:
        display_market_analysis(filtered_df)
    
    with tab3:
        display_financial_metrics(filtered_df)
    
    with tab4:
        display_property_details(filtered_df)
    
    with tab5:
        from dashboard.advanced_features import display_portfolio_analysis
        display_portfolio_analysis(filtered_df)
    
    with tab6:
        from dashboard.advanced_features import display_investment_recommendations
        display_investment_recommendations(filtered_df)
    
    with tab7:
        from dashboard.advanced_features import display_comparison_tool
        display_comparison_tool(filtered_df)


def display_deal_rankings(df: pd.DataFrame):
    """Display deal rankings and scores."""
    
    if df.empty:
        st.warning("No properties match the current filters.")
        return
    
    # Sort by deal score
    ranked_df = df.sort_values('deal_score', ascending=False).head(20)
    
    # Deal score distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df, 
            x='deal_score', 
            nbins=20,
            title="Deal Score Distribution",
            labels={'deal_score': 'Deal Score', 'count': 'Number of Properties'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cap rate vs Deal score scatter
        fig = px.scatter(
            df,
            x='cap_rate',
            y='deal_score',
            size='list_price',
            color='state',
            hover_data=['address', 'city'],
            title="Cap Rate vs Deal Score",
            labels={'cap_rate': 'Cap Rate (%)', 'deal_score': 'Deal Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top deals table
    st.subheader("🏆 Top Investment Opportunities")
    
    # Create a styled dataframe
    display_df = ranked_df[['address', 'city', 'state', 'list_price', 'cap_rate', 'cash_on_cash', 'deal_score']].copy()
    display_df['list_price'] = display_df['list_price'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
    display_df['cap_rate'] = display_df['cap_rate'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
    display_df['cash_on_cash'] = display_df['cash_on_cash'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
    display_df['deal_score'] = display_df['deal_score'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    
    st.dataframe(display_df, use_container_width=True)


def display_market_analysis(df: pd.DataFrame):
    """Display market analysis charts."""
    
    if df.empty:
        st.warning("No properties match the current filters.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution by city
        fig = px.box(
            df,
            x='city',
            y='list_price',
            title="Price Distribution by City"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cap rate by property type
        fig = px.box(
            df,
            x='property_type',
            y='cap_rate',
            title="Cap Rate by Property Type"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Market heatmap
    st.subheader("🔥 Market Heatmap")
    
    # Create heatmap data
    heatmap_data = df.groupby(['city', 'property_type'])['deal_score'].mean().unstack(fill_value=0)
    
    fig = px.imshow(
        heatmap_data,
        title="Average Deal Score by City and Property Type",
        labels=dict(x="Property Type", y="City", color="Deal Score"),
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)


def display_financial_metrics(df: pd.DataFrame):
    """Display financial metrics and analysis."""
    
    if df.empty:
        st.warning("No properties match the current filters.")
        return
    
    # Financial metrics overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_noi = df['noi'].mean()
        st.metric("Average NOI", f"${avg_noi:,.0f}" if not pd.isna(avg_noi) else "N/A")
    
    with col2:
        avg_cash_flow = df['monthly_cash_flow'].mean()
        st.metric("Avg Monthly Cash Flow", f"${avg_cash_flow:,.0f}" if not pd.isna(avg_cash_flow) else "N/A")
    
    with col3:
        avg_coc = df['cash_on_cash'].mean()
        st.metric("Avg Cash-on-Cash", f"{avg_coc:.1f}%" if not pd.isna(avg_coc) else "N/A")
    
    # Financial charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Cash flow distribution
        fig = px.histogram(
            df,
            x='monthly_cash_flow',
            nbins=20,
            title="Monthly Cash Flow Distribution",
            labels={'monthly_cash_flow': 'Monthly Cash Flow ($)', 'count': 'Number of Properties'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # NOI vs Price
        fig = px.scatter(
            df,
            x='list_price',
            y='noi',
            size='deal_score',
            color='state',
            hover_data=['address', 'city'],
            title="NOI vs Property Price",
            labels={'list_price': 'Property Price ($)', 'noi': 'Net Operating Income ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Income vs Expenses breakdown
    st.subheader("💰 Income vs Expenses Analysis")
    
    # Calculate averages
    avg_income = df['total_income'].mean()
    avg_expenses = df['total_expenses'].mean()
    avg_noi = df['noi'].mean()
    
    # Create waterfall chart
    fig = go.Figure(go.Waterfall(
        name="Financial Breakdown",
        orientation="h",
        measure=["relative", "relative", "total"],
        x=[avg_income, -avg_expenses, avg_noi],
        textposition="outside",
        text=[f"${avg_income:,.0f}", f"-${avg_expenses:,.0f}", f"${avg_noi:,.0f}"],
        y=["Gross Income", "Total Expenses", "Net Operating Income"],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="Average Financial Breakdown",
        showlegend=False,
        waterfallgap=0.2,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_property_details(df: pd.DataFrame):
    """Display detailed property information."""
    
    if df.empty:
        st.warning("No properties match the current filters.")
        return
    
    # Property search
    st.subheader("🔍 Property Search")
    
    search_term = st.text_input("Search by address or city:")
    
    if search_term:
        search_df = df[
            df['address'].str.contains(search_term, case=False, na=False) |
            df['city'].str.contains(search_term, case=False, na=False)
        ]
    else:
        search_df = df
    
    # Property details table
    if not search_df.empty:
        # Select columns to display
        display_columns = [
            'address', 'city', 'state', 'property_type', 'bedrooms', 'bathrooms',
            'square_feet', 'list_price', 'cap_rate', 'cash_on_cash', 'deal_score'
        ]
        
        display_df = search_df[display_columns].copy()
        
        # Format numeric columns
        display_df['list_price'] = display_df['list_price'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
        display_df['cap_rate'] = display_df['cap_rate'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        display_df['cash_on_cash'] = display_df['cash_on_cash'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        display_df['deal_score'] = display_df['deal_score'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Property details on click
        st.subheader("📋 Property Details")
        
        selected_property = st.selectbox(
            "Select a property for detailed analysis:",
            options=search_df['address'].tolist()
        )
        
        if selected_property:
            property_data = search_df[search_df['address'] == selected_property].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Property Information**")
                st.write(f"Address: {property_data['address']}")
                st.write(f"City: {property_data['city']}, {property_data['state']}")
                st.write(f"Property Type: {property_data['property_type']}")
                st.write(f"Bedrooms: {property_data['bedrooms']}")
                st.write(f"Bathrooms: {property_data['bathrooms']}")
                st.write(f"Square Feet: {property_data['square_feet']:,}" if pd.notna(property_data['square_feet']) else "Square Feet: N/A")
                st.write(f"Year Built: {property_data['year_built']}" if pd.notna(property_data['year_built']) else "Year Built: N/A")
            
            with col2:
                st.write("**Financial Metrics**")
                st.write(f"List Price: ₹{property_data['list_price']:,.0f}" if pd.notna(property_data['list_price']) else "List Price: N/A")
                st.write(f"Cap Rate: {property_data['cap_rate']:.1f}%" if pd.notna(property_data['cap_rate']) else "Cap Rate: N/A")
                st.write(f"Cash-on-Cash: {property_data['cash_on_cash']:.1f}%" if pd.notna(property_data['cash_on_cash']) else "Cash-on-Cash: N/A")
                st.write(f"Monthly Cash Flow: ${property_data['monthly_cash_flow']:,.0f}" if pd.notna(property_data['monthly_cash_flow']) else "Monthly Cash Flow: N/A")
                st.write(f"Deal Score: {property_data['deal_score']:.1f}" if pd.notna(property_data['deal_score']) else "Deal Score: N/A")
    
    else:
        st.info("No properties found matching your search criteria.")


if __name__ == "__main__":
    main()
