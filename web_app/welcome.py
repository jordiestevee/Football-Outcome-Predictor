import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Football Outcome Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .main-header h1 {
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        font-size: 1.3rem;
        margin-top: 1rem;
        opacity: 0.95;
    }
    .highlight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .dataset-info {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 2rem 0;
        color: black;
    }
    .nav-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        transition: transform 0.2s;
        cursor: pointer;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .nav-button:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    .comparison-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Main Header with Gradient
st.markdown("""
<div class="main-header">
    <h1>🏆 Predicting Football Match Outcomes with Explainable AI</h1>
    <p>⚽ Leveraging Machine Learning & XAI to Forecast Match Results with Transparency</p>
</div>
""", unsafe_allow_html=True)

# Dataset Introduction
st.markdown("""
<div class="dataset-info">
    <h2>📊 About the Dataset</h2>
    <p style="font-size: 1.1rem; line-height: 1.8; color: #333;">
    This project analyzes <strong>40,370 professional football matches</strong> from the <strong>top 8 European leagues</strong> 
    spanning <strong>15 years (2010–2025)</strong>. The dataset includes match results, team Elo ratings, recent form metrics, 
    goal statistics, head-to-head history, and betting odds from Bet365. Our comprehensive dataset covers leagues from 
    England (Premier League), Spain (La Liga), Italy (Serie A), Germany (Bundesliga), France (Ligue 1), 
    Netherlands (Eredivisie), Belgium (Pro League), and Portugal (Primeira Liga).
    </p>
</div>
""", unsafe_allow_html=True)

# Project Goal
st.markdown("""
<div class="highlight-box">
    <h2>🧠 Our Goal</h2>
    <p style="font-size: 1.2rem; margin-top: 1rem;">
    Build a <strong>probability-based model</strong> to forecast football match outcomes 
    (Home Win / Draw / Away Win) and <strong>explain predictions using Explainable AI (XAI)</strong> techniques, 
    specifically SHAP values, to provide transparent and interpretable insights into each prediction.
    </p>
</div>
""", unsafe_allow_html=True)

# Quick Stats
st.markdown("---")
st.markdown("## 📈 Dataset at a Glance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="⚽ Total Matches",
        value="40,370",
        delta="8 Top Leagues"
    )

with col2:
    st.metric(
        label="📅 Time Period",
        value="2010-2025",
        delta="15 Years"
    )

with col3:
    st.metric(
        label="🎯 Features",
        value="9",
        delta="Optimized Set"
    )

with col4:
    st.metric(
        label="🤖 Model Accuracy",
        value="52.96%",
        delta="vs Bet365: 54.24%"
    )


# Sample Predictions Table
st.markdown("### 🔮 Sample Model Predictions")

sample_predictions = pd.DataFrame({
    'Match': [
        'Manchester City vs Liverpool',
        'Barcelona vs Real Madrid',
        'Bayern Munich vs Borussia Dortmund',
        'PSG vs Marseille',
        'Inter Milan vs Juventus'
    ],
    'P(Home Win)': ['58.3%', '51.2%', '62.7%', '64.1%', '49.8%'],
    'P(Draw)': ['23.4%', '26.8%', '21.3%', '20.5%', '28.2%'],
    'P(Away Win)': ['18.3%', '22.0%', '16.0%', '15.4%', '22.0%'],
    'Prediction': ['Home Win', 'Home Win', 'Home Win', 'Home Win', 'Draw'],
    'Confidence': ['High (58%)', 'Medium (51%)', 'High (63%)', 'High (64%)', 'Low (50%)']
})

st.dataframe(sample_predictions, hide_index=True, use_container_width=True)

st.markdown("""
*Sample predictions demonstrate probability-based outputs with confidence levels. 
Each prediction includes the likelihood of all three outcomes.*
""")

# Navigation Section
st.markdown("---")
st.markdown("## 🚀 Explore the Application")

st.markdown("""
Navigate to different sections using the sidebar on the left, or click the buttons below:
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="nav-button">
        <h3>📊</h3>
        <h4>Data Exploration</h4>
        <p>Explore match statistics, trends, and patterns across leagues and seasons</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Data Exploration", use_container_width=True):
        st.switch_page("pages/1 Data Exploration.py")

with col2:
    st.markdown("""
    <div class="nav-button">
        <h3>🤖</h3>
        <h4>Match Forecast</h4>
        <p>Predict match outcomes with probability estimates and confidence levels</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Match Forecast", use_container_width=True):
        st.switch_page("pages/2 Match Forecast.py")

with col3:
    st.markdown("""
    <div class="nav-button">
        <h3>🔍</h3>
        <h4>Model Explainability</h4>
        <p>Understand how the model makes decisions using SHAP values</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Model Explainability", use_container_width=True):
        st.switch_page("pages/3 Model Explaniability.py")


# Project Methodology
st.markdown("---")
st.markdown("## 🔬 Project Methodology")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Data Collection")
    st.markdown("""
    - **40,370 matches** from football-data.co.uk
    - **8 top European leagues**
    - **15 years** of historical data
    - **9 engineered features** including Elo ratings, form, and odds
    """)

with col2:
    st.markdown("### 🤖 Model Training")
    st.markdown("""
    - **XGBoost** classifier (multi:softprob)
    - **80/20 train-test** split (time-based)
    - **Probability-based** predictions
    - **Hyperparameter tuning** for optimal performance
    """)

with col3:
    st.markdown("### 🔍 Explainability")
    st.markdown("""
    - **SHAP values** for feature attribution
    - **Class-specific** analysis (Home/Draw/Away)
    - **Local explanations** for individual predictions
    - **Global insights** for model behavior
    """)

# Team Information
st.markdown("---")
st.markdown("## 👥 Team Members")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 2rem; border-radius: 12px; text-align: center;">
        <h2>👤 Albert Jané Lardies</h2>
        <p style="font-size: 1.2rem;"><strong>NIA:</strong> 268537</p>
        <p style="font-size: 1.1rem;">Data Engineering & Model Development</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                color: white; padding: 2rem; border-radius: 12px; text-align: center;">
        <h2>👤 Jordi Esteve Claramunt</h2>
        <p style="font-size: 1.2rem;"><strong>NIA:</strong> 268829</p>
        <p style="font-size: 1.1rem;">Model Training & Web Application</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Football Outcome Predictor</strong> | Built using Streamlit</p>
</div>
""", unsafe_allow_html=True)
