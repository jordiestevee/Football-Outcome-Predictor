import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import shap
import sys
import os

# Page configuration
st.set_page_config(
    page_title="Match Forecast",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .prob-card-home {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(46, 204, 113, 0.3);
        transition: transform 0.3s;
    }
    .prob-card-draw {
        background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(149, 165, 166, 0.3);
        transition: transform 0.3s;
    }
    .prob-card-away {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(231, 76, 60, 0.3);
        transition: transform 0.3s;
    }
    .feature-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin: 1rem 0;
    }
    .similar-match {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 4px solid #3498db;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #2c3e50;
    }
    .insight-box p {
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    .insight-box h4 {
        color: #2c3e50;
        margin-top: 0;
    }
    .vs-divider {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;">
    <h1>🤖 Interactive Match Outcome Forecast</h1>
    <p style="font-size: 1.2rem; margin-top: 0.5rem;">
    Predict football match outcomes with probability-based machine learning
    </p>
</div>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    """Load trained XGBoost model from pickle file"""
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'model.pkl')
        with open(model_path, 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts['model']
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        return None

model_data = load_model()

# Load Elo ratings
@st.cache_data
def load_elo_ratings():
    """Load Elo ratings from CSV file"""
    try:
        df = pd.read_csv('../data/EloRatings.csv')
        # Get the most recent Elo for each team
        df['date'] = pd.to_datetime(df['date'])
        latest_elos = df.sort_values('date').groupby('club').last().reset_index()
        return latest_elos[['club', 'elo']].set_index('club')['elo'].to_dict()
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not load Elo ratings: {e}")
        return {}

def get_team_elo(team_name, elo_dict):
    """Get Elo rating for a team, with fallback to default"""
    if not elo_dict:
        return 1500
    
    # Direct match
    if team_name in elo_dict:
        return int(elo_dict[team_name])
    
    return 1500  # Default fallback

elo_ratings = load_elo_ratings()

# Load actual teams from dataset
@st.cache_data
def load_teams_from_dataset():
    """Load actual teams from Matches_Clean.csv"""
    try:
        df = pd.read_csv('../data/Matches_Clean.csv')
        
        # Get all unique teams
        all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
        all_teams = sorted(list(all_teams))
        
        # Map divisions to league names
        division_map = {
            'E0': 'Premier League',
            'SP1': 'La Liga',
            'I1': 'Serie A',
            'D1': 'Bundesliga',
            'F1': 'Ligue 1',
            'N1': 'Eredivisie',
            'B1': 'Pro League',
            'P1': 'Primeira Liga'
        }
        
        # Group teams by league
        teams_by_league = {league: [] for league in division_map.values()}
        
        for team in all_teams:
            # Find which league(s) this team plays in
            home_divisions = df[df['HomeTeam'] == team]['Division'].unique()
            away_divisions = df[df['AwayTeam'] == team]['Division'].unique()
            divisions = set(home_divisions) | set(away_divisions)
            
            for div in divisions:
                if div in division_map:
                    league = division_map[div]
                    if team not in teams_by_league[league]:
                        teams_by_league[league].append(team)
        
        # Sort teams within each league
        for league in teams_by_league:
            teams_by_league[league] = sorted(teams_by_league[league])
        
        return teams_by_league
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not load teams from dataset: {e}")
        return {}

TEAMS = load_teams_from_dataset()

# Sidebar - Model Information
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;">
    <h2>📊 Model Information</h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.info("""
**Algorithm:** XGBoost Classifier  
**Objective:** multi:softprob  
**Training Data:** 31,668 matches  
**Test Accuracy:** 52.96%  
**Log Loss:** 0.9826  

**Key Features:**
- Elo Difference
- Total Elo (Match Quality)
- Form5 Difference
- GF5 Difference
- GA5 Difference
- Last3 H2H

**Comparison:**
- Bet365 Accuracy: 54.24%
- Bet365 Log Loss: 0.9573
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 How to Use")
st.sidebar.markdown("""
1. **Select teams** from dropdowns
2. **Choose match date**
3. **Toggle neutral venue** if applicable
4. **Review feature values**
5. **Click PREDICT** to see probabilities
6. **Explore similar matches** from history
""")

# Main content area
st.markdown("---")
st.markdown("## ⚽ Match Configuration")

# Team selection with enhanced layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏠 Home Team")
    home_league = st.selectbox("Select Home League", list(TEAMS.keys()), key='home_league')
    home_team = st.selectbox("Select Home Team", TEAMS[home_league], key='home_team')

with col2:
    st.markdown("### ✈️ Away Team")
    away_league = st.selectbox("Select Away League", list(TEAMS.keys()), key='away_league')
    away_team = st.selectbox("Select Away Team", TEAMS[away_league], key='away_team')


# Feature Input Section
st.markdown("---")
st.markdown("## 📊 Feature Values")

if elo_ratings:
    st.success(f"✅ Loaded latest Elo ratings for {len(elo_ratings)} teams. Default values set automatically!")
else:
    st.info("💡 **Tip:** Adjust these values based on recent team performance, or use estimated values for future predictions.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏠 Home Team Stats")
    default_home_elo = get_team_elo(home_team, elo_ratings)
    home_elo = st.number_input(
        "Home Elo Rating", 
        min_value=1000, 
        max_value=2500, 
        value=default_home_elo, 
        step=10, 
        help=f"Latest Elo for {home_team}: {default_home_elo}"
    )
    home_form = st.number_input("Home Form Points (last 5)", min_value=0, max_value=15, value=7, step=1,
                                help="Points earned in last 5 matches (0-15)")
    home_gf = st.number_input("Home Goals Scored (last 5)", min_value=0, max_value=20, value=7, step=1,
                              help="Goals scored in last 5 matches")
    home_ga = st.number_input("Home Goals Conceded (last 5)", min_value=0, max_value=20, value=5, step=1,
                              help="Goals conceded in last 5 matches")

with col2:
    st.markdown("### ✈️ Away Team Stats")
    default_away_elo = get_team_elo(away_team, elo_ratings)
    away_elo = st.number_input(
        "Away Elo Rating", 
        min_value=1000, 
        max_value=2500, 
        value=default_away_elo, 
        step=10,
        help=f"Latest Elo for {away_team}: {default_away_elo}"
    )
    away_form = st.number_input("Away Form Points (last 5)", min_value=0, max_value=15, value=7, step=1,
                                help="Points earned in last 5 matches (0-15)")
    away_gf = st.number_input("Away Goals Scored (last 5)", min_value=0, max_value=20, value=5, step=1,
                              help="Goals scored in last 5 matches")
    away_ga = st.number_input("Away Goals Conceded (last 5)", min_value=0, max_value=20, value=7, step=1,
                              help="Goals conceded in last 5 matches")

# Head-to-head
st.markdown("### 🤝 Head-to-Head History")
h2h_value = st.number_input(
    "Goal Difference in Last 3 H2H Matches (Home perspective)", 
    min_value=-15, 
    max_value=15, 
    value=0, 
    step=1, 
    help="Positive = Home team scored more in recent H2H, Negative = Away team dominated"
)

# Calculate derived features
elo_diff = home_elo - away_elo 
total_elo = home_elo + away_elo
form_diff = home_form - away_form
gf_diff = home_gf - away_gf
ga_diff = home_ga - away_ga

# Display derived features
st.markdown("---")
st.markdown("## 🔬 Computed Model Features")


st.markdown("""
<div class="feature-box">
    <p style="margin: 0; font-size: 0.9em; color: #666;">
    These are the features the XGBoost model uses to make predictions. 
    They are derived from the input statistics above.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    delta_text = "Home advantage" if elo_diff > 0 else "Away advantage" if elo_diff < 0 else "Balanced"
    st.metric("Elo Difference", f"{elo_diff:+d}", delta=delta_text)

with col2:
    quality = "Elite" if total_elo > 3500 else "High" if total_elo > 3000 else "Average"
    st.metric("Total Elo", f"{total_elo}", delta=f"{quality} quality")

with col3:
    st.metric("Form Difference", f"{form_diff:+d}",
              delta="Home better" if form_diff > 0 else "Away better" if form_diff < 0 else "Equal")

with col4:
    st.metric("GF Difference", f"{gf_diff:+d}",
              delta="Home attack" if gf_diff > 0 else "Away attack" if gf_diff < 0 else "Equal")

with col5:
    st.metric("GA Difference", f"{ga_diff:+d}",
              delta="Home defense" if ga_diff < 0 else "Away defense" if ga_diff > 0 else "Equal")

with col6:
    st.metric("H2H", f"{h2h_value:+d}",
              delta="Home history" if h2h_value > 0 else "Away history" if h2h_value < 0 else "Neutral")

# Predict button
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button("🔮 PREDICT MATCH OUTCOME", use_container_width=True, type="primary", 
                                help="Click to generate probability-based prediction")

# Prediction logic
if predict_button:
    # Validation
    if home_team == away_team and home_league == away_league:
        st.error("❌ **Error:** Home and Away teams cannot be the same!")
        st.stop()
    
    # Create feature array exactly as in Model Explainability
    feature_values = np.array([elo_diff, total_elo, form_diff, gf_diff, ga_diff, h2h_value])
    
    # Make prediction exactly as in Model Explainability page
    if model_data is not None:
        try:
            prediction_proba = model_data.predict_proba(feature_values.reshape(1, -1))[0]
            prediction_class = np.argmax(prediction_proba)
            home_prob, draw_prob, away_prob = prediction_proba
        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")
            st.stop()
    else:
        st.error("❌ Model not loaded. Please ensure model.pkl is available.")
        st.stop()
    
    # Get prediction
    probs = np.array([home_prob, draw_prob, away_prob])
    prediction_idx = prediction_class
    prediction_labels = ['Home Win', 'Draw', 'Away Win']
    prediction = prediction_labels[prediction_idx]
    confidence = probs[prediction_idx]
    
    # Display prediction
    st.markdown("---")
    st.markdown("## 🎯 Prediction Results")
    
    st.markdown(f"""
    <div class="prediction-card">
        <h2>⚽ {home_team} vs {away_team}</h2>
        <p style="font-size: 1rem; opacity: 0.8; margin-top: 0.5rem;">🏠 Home: {home_team}</p>
    </div>
    """, unsafe_allow_html=True)
    # Animated probability cards
    st.markdown("### 📊 Predicted Probabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="prob-card-home">
            <h3 style="margin: 0;">🔵 Home Win</h3>
            <h1 style="margin: 1rem 0; font-size: 3rem;">{home_prob*100:.1f}%</h1>
            <p style="margin: 0; opacity: 0.9;">{home_team}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="prob-card-draw">
            <h3 style="margin: 0;">🟡 Draw</h3>
            <h1 style="margin: 1rem 0; font-size: 3rem;">{draw_prob*100:.1f}%</h1>
            <p style="margin: 0; opacity: 0.9;">Neither Team</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="prob-card-away">
            <h3 style="margin: 0;">🔴 Away Win</h3>
            <h1 style="margin: 1rem 0; font-size: 3rem;">{away_prob*100:.1f}%</h1>
            <p style="margin: 0; opacity: 0.9;">{away_team}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability visualization - Bar and Pie Charts
    st.markdown("---")
    st.markdown("### 📈 Probability Visualizations")
    
    col1, col2 = st.columns(2)
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        fig_bar = go.Figure()
        
        fig_bar.add_trace(go.Bar(
            x=prediction_labels,
            y=[home_prob*100, draw_prob*100, away_prob*100],
            marker_color=['#2ecc71', '#95a5a6', '#e74c3c'],
            text=[f"{home_prob*100:.1f}%", f"{draw_prob*100:.1f}%", f"{away_prob*100:.1f}%"],
            textposition='auto',
            textfont=dict(size=16, color='white', family='Arial Black'),
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>'
        ))
        
        fig_bar.update_layout(
            title="Probability Distribution (Bar Chart)",
            xaxis_title="Outcome",
            yaxis_title="Probability (%)",
            yaxis=dict(range=[0, 100]),
            height=400,
            showlegend=False
        )
        
        fig_bar.add_hline(y=33.33, line_dash="dash", line_color="red", opacity=0.5,
                      annotation_text="Random baseline (33.3%)")
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=prediction_labels,
            values=[home_prob*100, draw_prob*100, away_prob*100],
            marker_colors=['#2ecc71', '#95a5a6', '#e74c3c'],
            hole=0.4,
            textinfo='label+percent',
            textfont=dict(size=14, color='white'),
            hovertemplate='<b>%{label}</b><br>Probability: %{value:.1f}%<extra></extra>'
        )])
        
        fig_pie.update_layout(
            title="Probability Distribution (Pie Chart)",
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Confidence interpretation
    st.markdown("---")
    st.markdown("### 💬 Confidence Interpretation")
    
    # Generate intelligent interpretation
    max_prob = max(home_prob, draw_prob, away_prob)
    prob_diff = max_prob - sorted([home_prob, draw_prob, away_prob])[1]  # Difference to 2nd highest
    
    # Determine match type and confidence
    if draw_prob > 0.30:
        interpretation = f"""
        <div class="insight-box">
            <h4>⚖️ Balanced Match - High Draw Probability</h4>
            <p>The model sees <strong>equal momentum</strong> between both teams. 
            With a draw probability of <strong>{draw_prob*100:.1f}%</strong>, this match is 
            expected to be <strong>highly competitive</strong> with no clear favorite.</p>
            <p><strong>Key Insight:</strong> "Both teams are evenly matched across multiple dimensions."</p>
        </div>
        """
    elif prob_diff < 0.15:
        interpretation = f"""
        <div class="insight-box">
            <h4>🤝 Close Contest</h4>
            <p>This is a <strong>tight match</strong> with probabilities close together. 
            The model gives {prediction_labels[prediction_idx]} a slight edge at <strong>{max_prob*100:.1f}%</strong>, 
            but the outcome is <strong>highly uncertain</strong>.</p>
            <p><strong>Key Insight:</strong> Small changes in team form or tactics could swing the result.</p>
        </div>
        """
    elif max_prob > 0.60:
        favorite = prediction_labels[prediction_idx]
        interpretation = f"""
        <div class="insight-box">
            <h4>✅ Clear Favorite - {favorite}</h4>
            <p>The model is <strong>confident</strong> in predicting <strong>{favorite}</strong> with 
            <strong>{max_prob*100:.1f}%</strong> probability. This suggests a significant advantage 
            in key features like Elo rating, recent form, or historical performance.</p>
            <p><strong>Key Insight:</strong> The favorite has {prob_diff*100:.1f}% higher probability than the next most likely outcome.</p>
        </div>
        """
    else:
        interpretation = f"""
        <div class="insight-box">
            <h4>🎯 Moderate Prediction</h4>
            <p>The model predicts <strong>{prediction_labels[prediction_idx]}</strong> with 
            <strong>{max_prob*100:.1f}%</strong> confidence. While there's a preferred outcome, 
            other possibilities remain viable.</p>
            <p><strong>Key Insight:</strong> This is a competitive match with {prediction_labels[prediction_idx]} having a measurable but not overwhelming advantage.</p>
        </div>
        """
    
    st.markdown(interpretation, unsafe_allow_html=True)
    
    # Feature contribution using SHAP
    st.markdown("---")
    st.markdown("## 🔍 Feature Contributions")
    
    # Compute SHAP values for this specific prediction
    try:
        explainer_single = shap.TreeExplainer(model_data)
        shap_single = explainer_single(feature_values.reshape(1, -1))
        
        # Get SHAP contributions for the predicted class
        shap_contributions = shap_single.values[0][:, prediction_class]
        
        # Feature names
        feature_names = ['EloDifference', 'TotalElo', 'Form5Difference', 'GF5Difference', 'GA5Difference', 'Last3H2H']
        feature_values_display = [elo_diff, total_elo, form_diff, gf_diff, ga_diff, h2h_value]
        
        # Create dataframe
        df_impact = pd.DataFrame({
            'Feature': feature_names,
            'Value': feature_values_display,
            'SHAP': shap_contributions
        }).sort_values('SHAP', key=abs, ascending=True)
        
        # Visualize
        fig = go.Figure()
        
        colors = ['#2ecc71' if val > 0 else '#e74c3c' for val in df_impact['SHAP']]
        
        fig.add_trace(go.Bar(
            y=df_impact['Feature'],
            x=df_impact['SHAP'],
            orientation='h',
            marker_color=colors,
            text=[f"{val:+.3f}" for val in df_impact['SHAP']],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.4f}<br><extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Feature Contributions to {prediction} Probability",
            xaxis_title="SHAP Value (Impact on Prediction)",
            yaxis_title="",
            height=400,
            showlegend=False
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""<div class="insight-box">
        <p><strong>How to read this chart:</strong></p>
        <ul>
            <li><strong>Green bars</strong> = Features pushing <strong>towards</strong> {}</li>
            <li><strong>Red bars</strong> = Features pushing <strong>against</strong> {}</li>
            <li><strong>Longer bars</strong> = Stronger influence on the prediction</li>
        </ul>
        </div>""".format(prediction, prediction), unsafe_allow_html=True)
        
    except Exception as e:
        st.warning(f"⚠️ Could not compute SHAP values: {e}")
        st.info("Showing simplified feature importance instead.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🟢 Factors Favoring Home Win:")
        home_factors = []
        if elo_diff > 50:
            home_factors.append(f"✓ Strong Elo advantage (+{elo_diff})")
        if form_diff > 2:
            home_factors.append(f"✓ Better recent form (+{form_diff} points)")
        if gf_diff > 2:
            home_factors.append(f"✓ Superior attack (+{gf_diff} goals)")
        if h2h_value > 1:
            home_factors.append(f"✓ Positive H2H history (+{h2h_value})")
        
        if home_factors:
            for factor in home_factors:
                st.markdown(factor)
        else:
            st.markdown("*No strong home advantages detected*")
    
    with col2:
        st.markdown("#### 🔴 Factors Favoring Away Win:")
        away_factors = []
        if elo_diff < -50:
            away_factors.append(f"✓ Away team stronger Elo ({elo_diff})")
        if form_diff < -2:
            away_factors.append(f"✓ Away better form ({form_diff} points)")
        if gf_diff < -2:
            away_factors.append(f"✓ Away superior attack ({gf_diff} goals)")
        if h2h_value < -1:
            away_factors.append(f"✓ Away wins H2H history ({h2h_value})")
        
        if away_factors:
            for factor in away_factors:
                st.markdown(factor)
        else:
            st.markdown("*No strong away advantages detected*")
    

else:
    # Display placeholder when no prediction
    st.markdown("---")
    st.info("👆 Select teams and input their statistics, then click **PREDICT** to see the outcome forecast")
    
    # Show example prediction
    with st.expander("📖 See Example Prediction"):
        st.markdown("""
        ### Example: Manchester City (H) vs Arsenal (A)
        
        **Inputs:**
        - Home Elo: 2100, Away Elo: 2050
        - Home Form: 12, Away Form: 10
        - Home GF: 10, Away GF: 8
        - Home GA: 3, Away GA: 4
        - H2H: +2 (City advantage)
        
        **Prediction:**
        - 🏠 Home Win: **52.3%**
        - 🤝 Draw: **25.1%**
        - ✈️ Away Win: **22.6%**
        - Confidence: **MEDIUM** (52.3%)
        
        **Key Factors:**
        - Slight Elo advantage for home team
        - Better recent form
        - Stronger attack
        - Historical advantage
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><em>Match Forecast | Powered by XGBoost & Machine Learning</em></p>
</div>
""", unsafe_allow_html=True)
