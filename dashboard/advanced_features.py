"""
Advanced Dashboard Features for Real Estate Investment Analysis.
Includes portfolio analysis, risk assessment, and investment recommendations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any, Tuple
import json

from database.connection import db_manager
from database.models import Property, UnderwritingData, MLScore, TaxRecord, ZoningData


def display_portfolio_analysis(df: pd.DataFrame):
    """Display comprehensive portfolio analysis."""
    
    st.header("📊 Portfolio Analysis")
    
    if df.empty:
        st.warning("No properties available for portfolio analysis.")
        return
    
    # Portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_investment = df['list_price'].sum()
        st.metric("Total Portfolio Value", f"${total_investment:,.0f}")
    
    with col2:
        total_noi = df['noi'].sum()
        st.metric("Total Annual NOI", f"${total_noi:,.0f}")
    
    with col3:
        weighted_cap_rate = (df['noi'].sum() / df['list_price'].sum()) * 100
        st.metric("Weighted Cap Rate", f"{weighted_cap_rate:.1f}%")
    
    with col4:
        total_cash_flow = df['annual_cash_flow'].sum()
        st.metric("Total Annual Cash Flow", f"${total_cash_flow:,.0f}")
    
    # Portfolio composition
    col1, col2 = st.columns(2)
    
    with col1:
        # Property type distribution
        type_dist = df['property_type'].value_counts()
        fig = px.pie(
            values=type_dist.values,
            names=type_dist.index,
            title="Portfolio by Property Type"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Geographic distribution
        geo_dist = df.groupby(['state', 'city']).size().reset_index(name='count')
        fig = px.bar(
            geo_dist,
            x='city',
            y='count',
            color='state',
            title="Portfolio by Location"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk analysis
    st.subheader("🎯 Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk score distribution
        fig = px.histogram(
            df,
            x='risk_score',
            nbins=20,
            title="Risk Score Distribution",
            labels={'risk_score': 'Risk Score', 'count': 'Number of Properties'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk vs Return scatter
        fig = px.scatter(
            df,
            x='risk_score',
            y='cash_on_cash',
            size='list_price',
            color='deal_score',
            hover_data=['address', 'city'],
            title="Risk vs Return Analysis",
            labels={'risk_score': 'Risk Score', 'cash_on_cash': 'Cash-on-Cash Return (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)


def display_investment_recommendations(df: pd.DataFrame):
    """Display AI-powered investment recommendations."""
    
    st.header("🤖 AI Investment Recommendations")
    
    if df.empty:
        st.warning("No properties available for recommendations.")
        return
    
    # Calculate recommendation scores
    df_with_recommendations = calculate_recommendation_scores(df)
    
    # Top recommendations
    st.subheader("🏆 Top Investment Opportunities")
    
    top_recommendations = df_with_recommendations.nlargest(10, 'recommendation_score')
    
    for idx, row in top_recommendations.iterrows():
        with st.expander(f"#{row['recommendation_rank']}: {row['address']}, {row['city']}, {row['state']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Recommendation Score", f"{row['recommendation_score']:.1f}/100")
                st.metric("Deal Score", f"{row['deal_score']:.1f}")
                st.metric("Risk Score", f"{row['risk_score']:.1f}")
            
            with col2:
                st.metric("List Price", f"₹{row['list_price']:,.0f}")
                st.metric("Cap Rate", f"{row['cap_rate']:.1f}%")
                st.metric("Cash-on-Cash", f"{row['cash_on_cash']:.1f}%")
            
            with col3:
                st.metric("Monthly Cash Flow", f"${row['monthly_cash_flow']:,.0f}")
                st.metric("Annual NOI", f"${row['noi']:,.0f}")
                st.metric("ROI Potential", f"{row['roi_potential']:.1f}%")
            
            # Investment rationale
            st.write("**Investment Rationale:**")
            st.write(row['investment_rationale'])
            
            # Risk factors
            st.write("**Risk Factors:**")
            st.write(row['risk_factors'])


def calculate_recommendation_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive recommendation scores for properties."""
    
    df_copy = df.copy()
    
    # Normalize metrics to 0-100 scale
    metrics_to_normalize = [
        'deal_score', 'cap_rate', 'cash_on_cash', 'monthly_cash_flow'
    ]
    
    for metric in metrics_to_normalize:
        if metric in df_copy.columns and not df_copy[metric].isna().all():
            min_val = df_copy[metric].min()
            max_val = df_copy[metric].max()
            if max_val > min_val:
                df_copy[f'{metric}_normalized'] = (
                    (df_copy[metric] - min_val) / (max_val - min_val) * 100
                )
            else:
                df_copy[f'{metric}_normalized'] = 50  # Default to middle if no variation
    
    # Calculate ROI potential (simplified)
    # Ensure all values are numeric
    df_copy['cash_on_cash'] = pd.to_numeric(df_copy['cash_on_cash'], errors='coerce')
    df_copy['cap_rate'] = pd.to_numeric(df_copy['cap_rate'], errors='coerce')
    df_copy['risk_score'] = pd.to_numeric(df_copy['risk_score'], errors='coerce')
    
    df_copy['roi_potential'] = (
        df_copy['cash_on_cash'] + 
        (df_copy['cap_rate'] * 0.5) +  # Appreciation potential
        (100 - df_copy['risk_score']) * 0.3  # Risk-adjusted return
    )
    
    # Calculate recommendation score (weighted average)
    weights = {
        'deal_score_normalized': 0.3,
        'cap_rate_normalized': 0.2,
        'cash_on_cash_normalized': 0.2,
        'monthly_cash_flow_normalized': 0.15,
        'roi_potential': 0.15
    }
    
    df_copy['recommendation_score'] = 0.0  # Initialize as float
    for metric, weight in weights.items():
        if metric in df_copy.columns:
            # Ensure both operands are numeric
            df_copy[metric] = pd.to_numeric(df_copy[metric], errors='coerce')
            df_copy['recommendation_score'] += df_copy[metric] * weight
    
    # Ensure recommendation_score is float type
    df_copy['recommendation_score'] = pd.to_numeric(df_copy['recommendation_score'], errors='coerce')
    
    # Rank recommendations
    df_copy['recommendation_rank'] = df_copy['recommendation_score'].rank(ascending=False)
    
    # Generate investment rationale
    df_copy['investment_rationale'] = df_copy.apply(generate_investment_rationale, axis=1)
    df_copy['risk_factors'] = df_copy.apply(generate_risk_factors, axis=1)
    
    return df_copy


def generate_investment_rationale(row: pd.Series) -> str:
    """Generate investment rationale for a property."""
    
    rationale_parts = []
    
    # Deal score analysis
    if row['deal_score'] is not None:
        if row['deal_score'] >= 80:
            rationale_parts.append("Excellent deal score indicates strong investment potential.")
        elif row['deal_score'] >= 60:
            rationale_parts.append("Good deal score suggests solid investment opportunity.")
        else:
            rationale_parts.append("Moderate deal score - consider carefully.")
    else:
        rationale_parts.append("Deal score not available - unable to assess deal quality.")
    
    # Cap rate analysis
    if row['cap_rate'] is not None:
        if row['cap_rate'] >= 8:
            rationale_parts.append("High cap rate provides strong income potential.")
        elif row['cap_rate'] >= 6:
            rationale_parts.append("Competitive cap rate for the market.")
        else:
            rationale_parts.append("Lower cap rate but may offer appreciation potential.")
    else:
        rationale_parts.append("Cap rate not available - unable to assess income potential.")
    
    # Cash flow analysis
    if row['monthly_cash_flow'] is not None:
        if row['monthly_cash_flow'] > 0:
            rationale_parts.append("Positive cash flow provides immediate income.")
        else:
            rationale_parts.append("Negative cash flow - consider for appreciation potential only.")
    else:
        rationale_parts.append("Cash flow data not available - unable to assess income stream.")
    
    # Location analysis
    if row['city'] is not None:
        if row['city'] in ['Austin', 'Dallas', 'Houston']:
            rationale_parts.append("Strong Texas market with growing population.")
        elif row['city'] in ['Phoenix', 'Las Vegas']:
            rationale_parts.append("Sunbelt market with strong migration trends.")
    
    return " ".join(rationale_parts)


def generate_risk_factors(row: pd.Series) -> str:
    """Generate risk factors for a property."""
    
    risk_factors = []
    
    # Risk score analysis
    if row['risk_score'] is not None:
        if row['risk_score'] >= 70:
            risk_factors.append("High risk score - significant investment risk.")
        elif row['risk_score'] >= 50:
            risk_factors.append("Moderate risk - standard investment considerations apply.")
        else:
            risk_factors.append("Lower risk profile - relatively safer investment.")
    else:
        risk_factors.append("Risk score not available - unable to assess risk level.")
    
    # Property condition
    if 'condition' in row and row['condition']:
        if 'Needs Work' in str(row['condition']):
            risk_factors.append("Property may require significant repairs.")
        elif 'Fair' in str(row['condition']):
            risk_factors.append("Property condition may need attention.")
    
    # Market concentration
    if row['city'] in ['Austin', 'Dallas', 'Houston']:
        risk_factors.append("Texas market concentration - consider diversification.")
    
    # Price point risk
    if row['list_price'] > 1000000:
        risk_factors.append("High-value property - larger capital requirement.")
    elif row['list_price'] < 200000:
        risk_factors.append("Lower-value property - may have limited appreciation.")
    
    return " ".join(risk_factors) if risk_factors else "Standard investment risks apply."


def display_market_trends(df: pd.DataFrame):
    """Display market trend analysis."""
    
    st.header("📈 Market Trends Analysis")
    
    if df.empty:
        st.warning("No data available for market analysis.")
        return
    
    # Price trends by city
    st.subheader("🏙️ Price Trends by City")
    
    city_price_trends = df.groupby('city')['list_price'].agg(['mean', 'count']).reset_index()
    city_price_trends = city_price_trends.sort_values('mean', ascending=False)
    
    fig = px.bar(
        city_price_trends,
        x='city',
        y='mean',
        title="Average Property Prices by City",
        labels={'mean': 'Average Price ($)', 'city': 'City'}
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Cap rate trends
    st.subheader("💰 Cap Rate Trends")
    
    if 'cap_rate' in df.columns and not df['cap_rate'].isna().all():
        cap_rate_trends = df.groupby('city')['cap_rate'].mean().reset_index()
        cap_rate_trends = cap_rate_trends.sort_values('cap_rate', ascending=False)
        
        fig = px.bar(
            cap_rate_trends,
            x='city',
            y='cap_rate',
            title="Average Cap Rates by City",
            labels={'cap_rate': 'Cap Rate (%)', 'city': 'City'}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Market heatmap
    st.subheader("🔥 Market Heatmap")
    
    # Create heatmap data
    heatmap_data = df.groupby(['city', 'property_type']).agg({
        'deal_score': 'mean',
        'list_price': 'mean',
        'cap_rate': 'mean'
    }).reset_index()
    
    # Deal score heatmap
    deal_score_pivot = heatmap_data.pivot(index='city', columns='property_type', values='deal_score')
    
    fig = px.imshow(
        deal_score_pivot,
        title="Average Deal Score by City and Property Type",
        labels=dict(x="Property Type", y="City", color="Deal Score"),
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)


def display_property_details_enhanced(df: pd.DataFrame):
    """Display enhanced property details with tax and zoning information."""
    
    st.header("🏠 Enhanced Property Details")
    
    if df.empty:
        st.warning("No properties available for detailed view.")
        return
    
    # Property selector
    selected_property = st.selectbox(
        "Select a property for detailed analysis:",
        options=df['address'].tolist(),
        format_func=lambda x: f"{x}, {df[df['address']==x]['city'].iloc[0]}, {df[df['address']==x]['state'].iloc[0]}"
    )
    
    if selected_property:
        property_data = df[df['address'] == selected_property].iloc[0]
        
        # Property overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Property Overview")
            
            # Basic details
            details_data = {
                "Address": property_data['address'],
                "City": property_data['city'],
                "State": property_data['state'],
                "Property Type": property_data['property_type'],
                "Bedrooms": property_data['bedrooms'],
                "Bathrooms": property_data['bathrooms'],
                "Square Feet": f"{property_data['square_feet']:,}" if pd.notna(property_data['square_feet']) else "N/A",
                "Year Built": property_data['year_built'],
                "List Price": f"₹{property_data['list_price']:,.0f}" if pd.notna(property_data['list_price']) else "N/A"
            }
            
            for key, value in details_data.items():
                st.write(f"**{key}:** {value}")
        
        with col2:
            st.subheader("Investment Metrics")
            
            metrics_data = {
                "Deal Score": f"{property_data['deal_score']:.1f}",
                "Risk Score": f"{property_data['risk_score']:.1f}",
                "Cap Rate": f"{property_data['cap_rate']:.1f}%" if pd.notna(property_data['cap_rate']) else "N/A",
                "Cash-on-Cash": f"{property_data['cash_on_cash']:.1f}%" if pd.notna(property_data['cash_on_cash']) else "N/A",
                "Monthly Cash Flow": f"${property_data['monthly_cash_flow']:,.0f}" if pd.notna(property_data['monthly_cash_flow']) else "N/A"
            }
            
            for key, value in metrics_data.items():
                st.metric(key, value)
        
        # Financial breakdown
        st.subheader("💰 Financial Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Income vs Expenses
            if pd.notna(property_data['total_income']) and pd.notna(property_data['total_expenses']):
                fig = go.Figure(data=[
                    go.Bar(name='Income', x=['Total Income'], y=[property_data['total_income']], marker_color='green'),
                    go.Bar(name='Expenses', x=['Total Expenses'], y=[property_data['total_expenses']], marker_color='red'),
                    go.Bar(name='NOI', x=['Net Operating Income'], y=[property_data['noi']], marker_color='blue')
                ])
                fig.update_layout(title="Income vs Expenses", barmode='group')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cash flow breakdown
            if pd.notna(property_data['monthly_cash_flow']):
                fig = go.Figure(data=[
                    go.Bar(name='Monthly Cash Flow', x=['Cash Flow'], y=[property_data['monthly_cash_flow']], 
                           marker_color='green' if property_data['monthly_cash_flow'] > 0 else 'red')
                ])
                fig.update_layout(title="Monthly Cash Flow")
                st.plotly_chart(fig, use_container_width=True)
        
        # Tax and Zoning Information
        st.subheader("🏛️ Tax & Zoning Information")
        
        # Load additional data from database
        try:
            with db_manager.get_session() as session:
                property_id = property_data['id']
                
                # Tax information
                tax_record = session.query(TaxRecord).filter(TaxRecord.property_id == property_id).first()
                if tax_record:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Tax Assessment:**")
                        st.write(f"Assessed Value: ${tax_record.assessed_value:,.0f}" if tax_record.assessed_value else "N/A")
                        st.write(f"Annual Taxes: ${tax_record.annual_taxes:,.0f}" if tax_record.annual_taxes else "N/A")
                        st.write(f"Tax Rate: {tax_record.tax_rate:.2%}" if tax_record.tax_rate else "N/A")
                    
                    with col2:
                        st.write("**Land Values:**")
                        st.write(f"Land Value: ${tax_record.land_value:,.0f}" if tax_record.land_value else "N/A")
                        st.write(f"Improvement Value: ${tax_record.improvement_value:,.0f}" if tax_record.improvement_value else "N/A")
                
                # Zoning information
                zoning_record = session.query(ZoningData).filter(ZoningData.property_id == property_id).first()
                if zoning_record:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Zoning Information:**")
                        st.write(f"Zoning Code: {zoning_record.zoning_code}")
                        st.write(f"Land Use: {zoning_record.land_use}")
                        st.write(f"Max Density: {zoning_record.max_density} units/acre" if zoning_record.max_density else "N/A")
                        st.write(f"Max Height: {zoning_record.max_height} ft" if zoning_record.max_height else "N/A")
                    
                    with col2:
                        st.write("**Development Potential:**")
                        if zoning_record.setback_requirements:
                            try:
                                setbacks = json.loads(zoning_record.setback_requirements) if isinstance(zoning_record.setback_requirements, str) else zoning_record.setback_requirements
                                st.write("Setback Requirements:")
                                for setback_type, value in setbacks.items():
                                    st.write(f"  {setback_type.title()}: {value} ft")
                            except:
                                st.write("Setback data unavailable")
                        
                        if zoning_record.permitted_uses:
                            try:
                                permitted = json.loads(zoning_record.permitted_uses) if isinstance(zoning_record.permitted_uses, str) else zoning_record.permitted_uses
                                st.write("Permitted Uses:")
                                for use in permitted[:3]:  # Show first 3
                                    st.write(f"  • {use}")
                                if len(permitted) > 3:
                                    st.write(f"  ... and {len(permitted) - 3} more")
                            except:
                                st.write("Permitted uses data unavailable")
        
        except Exception as e:
            st.error(f"Error loading additional property data: {e}")


def display_comparison_tool(df: pd.DataFrame):
    """Display property comparison tool."""
    
    st.header("⚖️ Property Comparison Tool")
    
    if df.empty:
        st.warning("No properties available for comparison.")
        return
    
    # Property selection
    col1, col2 = st.columns(2)
    
    with col1:
        property1 = st.selectbox(
            "Select first property:",
            options=df['address'].tolist(),
            key="property1"
        )
    
    with col2:
        property2 = st.selectbox(
            "Select second property:",
            options=df['address'].tolist(),
            key="property2"
        )
    
    if property1 and property2 and property1 != property2:
        prop1_data = df[df['address'] == property1].iloc[0]
        prop2_data = df[df['address'] == property2].iloc[0]
        
        # Comparison table
        st.subheader("Property Comparison")
        
        comparison_data = {
            "Address": [prop1_data['address'], prop2_data['address']],
            "City": [prop1_data['city'], prop2_data['city']],
            "Property Type": [prop1_data['property_type'], prop2_data['property_type']],
            "List Price": [f"₹{prop1_data['list_price']:,.0f}", f"₹{prop2_data['list_price']:,.0f}"],
            "Deal Score": [f"{prop1_data['deal_score']:.1f}", f"{prop2_data['deal_score']:.1f}"],
            "Risk Score": [f"{prop1_data['risk_score']:.1f}", f"{prop2_data['risk_score']:.1f}"],
            "Cap Rate": [f"{prop1_data['cap_rate']:.1f}%", f"{prop2_data['cap_rate']:.1f}%"],
            "Cash-on-Cash": [f"{prop1_data['cash_on_cash']:.1f}%", f"{prop2_data['cash_on_cash']:.1f}%"],
            "Monthly Cash Flow": [f"${prop1_data['monthly_cash_flow']:,.0f}", f"${prop2_data['monthly_cash_flow']:,.0f}"]
        }
        
        comparison_df = pd.DataFrame(comparison_data, index=["Property 1", "Property 2"])
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visual comparison
        st.subheader("Visual Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Deal score comparison
            fig = px.bar(
                x=["Property 1", "Property 2"],
                y=[prop1_data['deal_score'], prop2_data['deal_score']],
                title="Deal Score Comparison",
                labels={'x': 'Property', 'y': 'Deal Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cap rate comparison
            fig = px.bar(
                x=["Property 1", "Property 2"],
                y=[prop1_data['cap_rate'], prop2_data['cap_rate']],
                title="Cap Rate Comparison",
                labels={'x': 'Property', 'y': 'Cap Rate (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Investment recommendation
        st.subheader("Investment Recommendation")
        
        score1 = prop1_data['deal_score'] - prop1_data['risk_score']
        score2 = prop2_data['deal_score'] - prop2_data['risk_score']
        
        if score1 > score2:
            st.success(f"**Recommendation: Property 1** ({prop1_data['address']})")
            st.write(f"Property 1 has a better risk-adjusted return with a score of {score1:.1f} vs {score2:.1f} for Property 2.")
        elif score2 > score1:
            st.success(f"**Recommendation: Property 2** ({prop2_data['address']})")
            st.write(f"Property 2 has a better risk-adjusted return with a score of {score2:.1f} vs {score1:.1f} for Property 1.")
        else:
            st.info("Both properties have similar risk-adjusted returns. Consider other factors like location preference and investment timeline.")
