import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path to import data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Data Exploration",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 4px solid #3498db;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #2c3e50;
    }
    .insight-box h4 {
        color: #2c3e50;
        margin-top: 0;
    }
    .insight-box p {
        color: #2c3e50;
    }
    .stat-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        margin: 0.5rem 0;
        color: #2c3e50;
    }
    .stat-box h4 {
        color: #2c3e50;
        margin: 0 0 0.5rem 0;
    }
    .stat-box p {
        color: #2c3e50;
        margin: 0.25rem 0;
    }
    .highlight {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;">
    <h1>📊 Interactive Data Exploration</h1>
    <p style="font-size: 1.2rem; margin-top: 0.5rem;">
    Deep dive into 40,000+ football matches with visual analytics
    </p>
</div>
""", unsafe_allow_html=True)

# Load data (or use sample data for demo)
@st.cache_data
def load_data():
    """Load cleaned match data from CSV with all engineered features"""
    try:
        # Load pre-processed data with engineered features
        df = pd.read_csv('../data/Matches_Clean.csv')
        df['MatchDate'] = pd.to_datetime(df['MatchDate'])
        df['Season'] = df['MatchDate'].dt.year
        
        # Map league names from Division code
        league_names = {
            'E0': 'Premier League',
            'SP1': 'La Liga',
            'I1': 'Serie A',
            'D1': 'Bundesliga',
            'F1': 'Ligue 1',
            'N1': 'Eredivisie',
            'B1': 'Pro League',
            'P1': 'Primeira Liga'
        }
        df['LeagueName'] = df['Division'].map(league_names)
        
        # Ensure all required features exist
        required_features = ['EloDifference', 'TotalElo', 'Form5Difference', 
                           'GF5Difference', 'GA5Difference', 'Last3H2H']
        for feature in required_features:
            if feature not in df.columns:
                st.warning(f"⚠️ Feature '{feature}' not found in CSV. Adding as placeholder.")
                df[feature] = 0
        
        return df
        
    except FileNotFoundError:
        st.error("⚠️ Could not find Matches_Clean.csv file. Please ensure it's in the data/ folder.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading data: {str(e)}")
        st.stop()

df = load_data()

# Sidebar filters
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;">
    <h2>🔍 Interactive Filters</h2>
</div>
""", unsafe_allow_html=True)

# League filter with visual separator
st.sidebar.markdown("### 🏆 Leagues")
selected_leagues = st.sidebar.multiselect(
    "Select one or more leagues:",
    options=sorted([x for x in df['LeagueName'].unique() if pd.notna(x)]),
    default=sorted([x for x in df['LeagueName'].unique() if pd.notna(x)]),
    help="Filter matches by specific leagues"
)

st.sidebar.markdown("---")

# Season filter
st.sidebar.markdown("### 📅 Seasons")
min_year = int(df['Season'].min())
max_year = int(df['Season'].max())

season_options = list(range(min_year, max_year + 1))
selected_seasons = st.sidebar.multiselect(
    "Select season(s):",
    options=season_options,
    default=season_options,
    help="Filter by specific seasons"
)

st.sidebar.markdown("---")

# Outcome filter
st.sidebar.markdown("### ⚽ Match Outcomes")
outcome_filter = st.sidebar.multiselect(
    "Filter by result:",
    options=['Home Win', 'Draw', 'Away Win'],
    default=['Home Win', 'Draw', 'Away Win'],
    help="Show only specific match outcomes"
)

st.sidebar.markdown("---")

# Advanced filters
with st.sidebar.expander("🎯 Advanced Filters"):
    elo_range = st.slider(
        "Elo Difference Range",
        min_value=float(df['EloDifference'].min()),
        max_value=float(df['EloDifference'].max()),
        value=(float(df['EloDifference'].min()), float(df['EloDifference'].max())),
        help="Filter by home team Elo advantage"
    )
    
    total_elo_range = st.slider(
        "Total Elo (Match Quality)",
        min_value=float(df['TotalElo'].min()),
        max_value=float(df['TotalElo'].max()),
        value=(float(df['TotalElo'].min()), float(df['TotalElo'].max())),
        help="Higher = better quality match"
    )

# Map outcome filter
outcome_map = {'Home Win': 'H', 'Draw': 'D', 'Away Win': 'A'}
selected_outcomes = [outcome_map[o] for o in outcome_filter]

# Apply filters
df_filtered = df[
    (df['LeagueName'].isin(selected_leagues)) &
    (df['Season'].isin(selected_seasons)) &
    (df['FTResult'].isin(selected_outcomes)) &
    (df['EloDifference'].between(elo_range[0], elo_range[1])) &
    (df['TotalElo'].between(total_elo_range[0], total_elo_range[1]))
]

# Sidebar summary
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Filter Summary")
st.sidebar.info(f"""
**Matches Selected:** {len(df_filtered):,} / {len(df):,}  
**Percentage:** {len(df_filtered)/len(df)*100:.1f}%  
**Leagues:** {len(selected_leagues)}  
**Seasons:** {len(selected_seasons)}
""")

# Dataset Overview with Enhanced Metrics
st.markdown("## 📈 Summary Statistics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3 style="margin: 0; font-size: 2rem;">⚽</h3>
        <h2 style="margin: 0.5rem 0;">{:,}</h2>
        <p style="margin: 0; opacity: 0.9;">Total Matches</p>
    </div>
    """.format(len(df_filtered)), unsafe_allow_html=True)

with col2:
    home_wins = (df_filtered['FTResult'] == 'H').sum()
    home_pct = home_wins/len(df_filtered)*100 if len(df_filtered) > 0 else 0
    st.markdown("""
    <div class="metric-card" style="background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);">
        <h3 style="margin: 0; font-size: 2rem;">🏠</h3>
        <h2 style="margin: 0.5rem 0;">{:,}</h2>
        <p style="margin: 0; opacity: 0.9;">Home Wins ({:.1f}%)</p>
    </div>
    """.format(home_wins, home_pct), unsafe_allow_html=True)

with col3:
    draws = (df_filtered['FTResult'] == 'D').sum()
    draw_pct = draws/len(df_filtered)*100 if len(df_filtered) > 0 else 0
    st.markdown("""
    <div class="metric-card" style="background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%);">
        <h3 style="margin: 0; font-size: 2rem;">🤝</h3>
        <h2 style="margin: 0.5rem 0;">{:,}</h2>
        <p style="margin: 0; opacity: 0.9;">Draws ({:.1f}%)</p>
    </div>
    """.format(draws, draw_pct), unsafe_allow_html=True)

with col4:
    away_wins = (df_filtered['FTResult'] == 'A').sum()
    away_pct = away_wins/len(df_filtered)*100 if len(df_filtered) > 0 else 0
    st.markdown("""
    <div class="metric-card" style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);">
        <h3 style="margin: 0; font-size: 2rem;">✈️</h3>
        <h2 style="margin: 0.5rem 0;">{:,}</h2>
        <p style="margin: 0; opacity: 0.9;">Away Wins ({:.1f}%)</p>
    </div>
    """.format(away_wins, away_pct), unsafe_allow_html=True)

with col5:
    avg_elo_diff = df_filtered['EloDifference'].mean()
    st.markdown("""
    <div class="metric-card" style="background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);">
        <h3 style="margin: 0; font-size: 2rem;">📊</h3>
        <h2 style="margin: 0.5rem 0;">{:+.0f}</h2>
        <p style="margin: 0; opacity: 0.9;">Avg Elo Diff</p>
    </div>
    """.format(avg_elo_diff), unsafe_allow_html=True)

# Match Outcome Distribution - Enhanced
st.markdown("---")
st.markdown("## ⚽ Outcome Distribution Analysis")

outcome_map_reverse = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}

col1, col2 = st.columns([3, 2])

with col1:
    # Interactive bar chart with comparison
    outcome_counts = df_filtered['FTResult'].value_counts()
    outcome_data = pd.DataFrame({
        'Outcome': [outcome_map_reverse[k] for k in ['H', 'D', 'A']],
        'Count': [outcome_counts.get(k, 0) for k in ['H', 'D', 'A']],
        'Percentage': [outcome_counts.get(k, 0) / len(df_filtered) * 100 if len(df_filtered) > 0 else 0 for k in ['H', 'D', 'A']]
    })
    
    fig = go.Figure()
    
    # Actual distribution
    fig.add_trace(go.Bar(
        name='Actual',
        x=outcome_data['Outcome'],
        y=outcome_data['Percentage'],
        marker_color=['#2ecc71', '#95a5a6', '#e74c3c'],
        text=[f"{p:.1f}%" for p in outcome_data['Percentage']],
        textposition='auto',
        textfont=dict(size=14, color='white', family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>Percentage: %{y:.1f}%<br>Count: %{customdata}<extra></extra>',
        customdata=outcome_data['Count']
    ))
    
    # Expected distribution (random)
    fig.add_trace(go.Bar(
        name='Expected (Random)',
        x=outcome_data['Outcome'],
        y=[33.33, 33.33, 33.33],
        marker_color='rgba(200, 200, 200, 0.3)',
        marker_line=dict(color='red', width=2),
        hovertemplate='<b>%{x}</b><br>Expected: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Match Outcome Distribution vs Random Baseline",
        xaxis_title="Outcome",
        yaxis_title="Percentage (%)",
        height=450,
        barmode='group',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 📊 Statistical Insights")
    
    if len(df_filtered) > 0:
        home_pct = (df_filtered['FTResult'] == 'H').sum() / len(df_filtered) * 100
        draw_pct = (df_filtered['FTResult'] == 'D').sum() / len(df_filtered) * 100
        away_pct = (df_filtered['FTResult'] == 'A').sum() / len(df_filtered) * 100
        expected_pct = 33.33
        
        home_advantage = home_pct - expected_pct
        draw_deficit = draw_pct - expected_pct
        away_deficit = away_pct - expected_pct
        
        # Determine colors for each outcome
        home_color = '#2ecc71' if home_advantage > 0 else '#e74c3c'
        draw_color = '#2ecc71' if draw_deficit > 0 else '#e74c3c'
        away_color = '#2ecc71' if away_deficit > 0 else '#e74c3c'
        
        # Build HTML content for insight box
        home_color_hex = '#2ecc71' if home_advantage > 0 else '#e74c3c'
        draw_color_hex = '#2ecc71' if draw_deficit > 0 else '#e74c3c'
        away_color_hex = '#2ecc71' if away_deficit > 0 else '#e74c3c'
        
        html_content = f"""
        <div class="insight-box">
            <h4>🏠 Home Advantage Effect</h4>
            <p><strong>Home:</strong> {home_pct:.1f}% <span style="font-weight: bold; color: {home_color_hex};">({home_advantage:+.1f}% vs random)</span></p>
            <p><strong>Draw:</strong> {draw_pct:.1f}% <span style="font-weight: bold; color: {draw_color_hex};">({draw_deficit:+.1f}% vs random)</span></p>
            <p><strong>Away:</strong> {away_pct:.1f}% <span style="font-weight: bold; color: {away_color_hex};">({away_deficit:+.1f}% vs random)</span></p>
        </div>
        """
        st.markdown(html_content, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight">
            <h4 style="margin-top: 0;">💡 Key Finding</h4>
            <p style="margin-bottom: 0;">Home teams win significantly more than expected by chance, 
            demonstrating a strong home advantage effect in professional football.</p>
        </div>
        """, unsafe_allow_html=True)

# Elo Difference vs Outcome - Key Visual Analytics
st.markdown("---")
st.markdown("## 📊 Elo Difference vs Match Outcome")

tab1, tab2 = st.tabs(["📈 Violin Plot", "🎯 Scatter Plot"])

with tab1:
    # Enhanced violin plot
    df_filtered['OutcomeLabel'] = df_filtered['FTResult'].map(outcome_map_reverse)
    
    fig = go.Figure()
    
    for outcome, color, label in [('H', '#2ecc71', 'Home Win'), ('D', '#95a5a6', 'Draw'), ('A', '#e74c3c', 'Away Win')]:
        data_subset = df_filtered[df_filtered['FTResult'] == outcome]['EloDifference']
        
        fig.add_trace(go.Violin(
            x=[label] * len(data_subset),
            y=data_subset,
            name=label,
            box_visible=True,
            meanline_visible=True,
            fillcolor=color,
            opacity=0.6,
            line_color=color,
            hovertemplate='<b>%{x}</b><br>Elo Diff: %{y:.1f}<extra></extra>'
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", 
                  annotation_text="Balanced Teams (Elo Diff = 0)")
    
    fig.update_layout(
        title="Elo Difference Distribution by Match Outcome",
        xaxis_title="Match Outcome",
        yaxis_title="Elo Difference (Home - Away)",
        height=500,
        showlegend=False,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        home_elo_avg = df_filtered[df_filtered['FTResult'] == 'H']['EloDifference'].mean()
        st.markdown(f"""
        <div class="stat-box" style="border-left: 4px solid #2ecc71;">
            <h4>🏠 Home Wins</h4>
            <p><strong>Avg Elo Diff:</strong> {home_elo_avg:+.1f}</p>
            <p style="font-size: 0.9em; color: #666;">Home teams win when stronger on average</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        draw_elo_avg = df_filtered[df_filtered['FTResult'] == 'D']['EloDifference'].mean()
        st.markdown(f"""
        <div class="stat-box" style="border-left: 4px solid #95a5a6;">
            <h4>🤝 Draws</h4>
            <p><strong>Avg Elo Diff:</strong> {draw_elo_avg:+.1f}</p>
            <p style="font-size: 0.9em; color: #666;">Draws occur with balanced teams</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        away_elo_avg = df_filtered[df_filtered['FTResult'] == 'A']['EloDifference'].mean()
        st.markdown(f"""
        <div class="stat-box" style="border-left: 4px solid #e74c3c;">
            <h4>✈️ Away Wins</h4>
            <p><strong>Avg Elo Diff:</strong> {away_elo_avg:+.1f}</p>
            <p style="font-size: 0.9em; color: #666;">Away teams win when significantly stronger</p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    # Scatter plot with jitter
    sample_size = min(2000, len(df_filtered))
    df_sample = df_filtered.sample(n=sample_size, random_state=42) if len(df_filtered) > sample_size else df_filtered
    
    fig = px.scatter(
        df_sample,
        x='EloDifference',
        y='Form5Difference',
        color='OutcomeLabel',
        color_discrete_map={'Home Win': '#2ecc71', 'Draw': '#95a5a6', 'Away Win': '#e74c3c'},
        title=f"Elo Difference vs Form Difference (Sample: {sample_size:,} matches)",
        labels={'EloDifference': 'Elo Difference (Home - Away)', 
                'Form5Difference': 'Form Difference (Last 5 matches)',
                'OutcomeLabel': 'Outcome'},
        opacity=0.6,
        height=500
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(hovermode='closest')
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **💡 Insight:** Notice the quadrant patterns:
    - **Top-right** (positive Elo + Form): Strong home win region
    - **Bottom-left** (negative Elo + Form): Strong away win region  
    - **Mixed quadrants**: More competitive matches with higher draw probability
    """)

# Temporal Trends - Enhanced with Draw Rate Analysis
st.markdown("---")
st.markdown("## 📅 Temporal Trends & Patterns")

# Calculate yearly statistics
yearly_outcomes = df_filtered.groupby(['Season', 'FTResult']).size().reset_index(name='Count')
yearly_totals = df_filtered.groupby('Season').size().reset_index(name='Total')
yearly_stats = yearly_outcomes.merge(yearly_totals, on='Season')
yearly_stats['Percentage'] = yearly_stats['Count'] / yearly_stats['Total'] * 100
yearly_stats['Outcome'] = yearly_stats['FTResult'].map(outcome_map_reverse)

col1, col2 = st.columns(2)

with col1:
    # Multi-line trend chart for all outcomes
    fig = go.Figure()
    
    for outcome, color, label in [('H', '#2ecc71', 'Home Win'), ('D', '#95a5a6', 'Draw'), ('A', '#e74c3c', 'Away Win')]:
        data = yearly_stats[yearly_stats['FTResult'] == outcome]
        fig.add_trace(go.Scatter(
            x=data['Season'],
            y=data['Percentage'],
            mode='lines+markers',
            name=label,
            line=dict(color=color, width=3),
            marker=dict(size=8, symbol='circle'),
            hovertemplate='<b>%{fullData.name}</b><br>Season: %{x}<br>Rate: %{y:.1f}%<extra></extra>'
        ))
    
    fig.add_hline(y=33.33, line_dash="dash", line_color="red", 
                  opacity=0.5, annotation_text="Random (33.3%)", annotation_position="left")
    
    fig.update_layout(
        title="Outcome Rates Over Time (Trend Analysis)",
        xaxis_title="Season",
        yaxis_title="Outcome Rate (%)",
        height=450,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Draw rate focused analysis
    draw_data = yearly_stats[yearly_stats['FTResult'] == 'D'][['Season', 'Percentage']].copy()
    
    if len(draw_data) > 0:
        # Add trend line
        from numpy.polynomial import Polynomial
        if len(draw_data) >= 2:
            p = Polynomial.fit(draw_data['Season'], draw_data['Percentage'], 1)
            draw_data['Trend'] = p(draw_data['Season'])
        
        fig = go.Figure()
        
        # Actual draw rate
        fig.add_trace(go.Scatter(
            x=draw_data['Season'],
            y=draw_data['Percentage'],
            mode='lines+markers',
            name='Actual Draw Rate',
            line=dict(color='#95a5a6', width=3),
            marker=dict(size=10, color='#95a5a6'),
            fill='tozeroy',
            fillcolor='rgba(149, 165, 166, 0.2)',
            hovertemplate='<b>Draw Rate</b><br>Season: %{x}<br>Rate: %{y:.1f}%<extra></extra>'
        ))
        
        # Trend line
        if len(draw_data) >= 2:
            fig.add_trace(go.Scatter(
                x=draw_data['Season'],
                y=draw_data['Trend'],
                mode='lines',
                name='Trend Line',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='<b>Trend</b><br>Season: %{x}<br>%{y:.1f}%<extra></extra>'
            ))
        
        fig.add_hline(y=33.33, line_dash="dot", line_color="orange", 
                      opacity=0.5, annotation_text="Expected (33.3%)")
        
        fig.update_layout(
            title="⚡ Draw Rate Per Season (Key Insight)",
            xaxis_title="Season",
            yaxis_title="Draw Rate (%)",
            height=450,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Statistical insights for trends
st.markdown("### 📊 Temporal Insights")

col1, col2, col3 = st.columns(3)

if len(draw_data) > 0:
    min_draw_season = draw_data.loc[draw_data['Percentage'].idxmin(), 'Season']
    min_draw_rate = draw_data['Percentage'].min()
    max_draw_season = draw_data.loc[draw_data['Percentage'].idxmax(), 'Season']
    max_draw_rate = draw_data['Percentage'].max()
    avg_draw_rate = draw_data['Percentage'].mean()
    
    with col1:
        st.markdown(f"""
        <div class="insight-box">
            <h4>📉 Lowest Draw Rate</h4>
            <p><strong>Season:</strong> {min_draw_season:.0f}</p>
            <p><strong>Rate:</strong> {min_draw_rate:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="insight-box">
            <h4>📈 Highest Draw Rate</h4>
            <p><strong>Season:</strong> {max_draw_season:.0f}</p>
            <p><strong>Rate:</strong> {max_draw_rate:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="insight-box">
            <h4>📊 Average Draw Rate</h4>
            <p><strong>Overall:</strong> {avg_draw_rate:.1f}%</p>
            <p><strong>Variance:</strong> {draw_data['Percentage'].std():.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

# Feature Analysis - Enhanced
st.markdown("---")
st.markdown("## 🔍 Advanced Feature Analysis")

feature_tab1, feature_tab2, feature_tab3 = st.tabs(["📊 Distributions", "🔗 Correlation Heatmap", "📈 Impact Analysis"])

with feature_tab1:
    st.markdown("### Feature Distributions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.histogram(
            df_filtered,
            x='EloDifference',
            nbins=50,
            title="Elo Difference Distribution",
            labels={'EloDifference': 'Elo Difference (Home - Away)'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            df_filtered,
            x='Form5Difference',
            nbins=30,
            title="Form Difference Distribution",
            labels={'Form5Difference': 'Form Difference (Last 5)'},
            color_discrete_sequence=['#2ecc71']
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.histogram(
            df_filtered,
            x='Last3H2H',
            nbins=20,
            title="Head-to-Head Distribution",
            labels={'Last3H2H': 'H2H Goal Difference (Last 3)'},
            color_discrete_sequence=['#9b59b6']
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

with feature_tab2:
    st.markdown("### 🔥 Correlation Heatmap")
    
    # Enhanced correlation heatmap
    features = ['EloDifference', 'TotalElo', 'Form5Difference', 'GF5Difference', 'GA5Difference', 'Last3H2H']
    corr_matrix = df_filtered[features].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_masked = corr_matrix.copy()
    corr_masked[mask] = np.nan
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_masked.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_masked.values, 2),
        texttemplate='%{text}',
        textfont={"size": 14, "color": "white"},
        colorbar=dict(
            title=dict(text="Correlation<br>Coefficient", side="right"),
            tickmode="linear",
            tick0=-1,
            dtick=0.5
        ),
        hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix (Lower Triangle)",
        height=600,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Find strongest correlations
    corr_pairs = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            corr_pairs.append({
                'Feature 1': features[i],
                'Feature 2': features[j],
                'Correlation': corr_matrix.iloc[i, j]
            })
    
    corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False).head(5)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="insight-box">
            <h4>💡 Key Correlation Insights</h4>
            <ul>
                <li><strong>Low multicollinearity:</strong> No correlation > 0.7, features are relatively independent</li>
                <li><strong>Form & Goals:</strong> Positive correlation between form and goal-scoring metrics</li>
                <li><strong>Defense & Attack:</strong> Teams strong offensively tend to be strong defensively</li>
                <li><strong>Independence:</strong> Good for machine learning (XGBoost can use all features effectively)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Top Correlations")
        for _, row in corr_df.iterrows():
            color = '#2ecc71' if row['Correlation'] > 0 else '#e74c3c'
            st.markdown(f"""
            <div class="stat-box">
                <p style="margin: 0;"><strong>{row['Feature 1'][:12]}</strong> ↔️ <strong>{row['Feature 2'][:12]}</strong></p>
                <p style="margin: 0.5rem 0 0 0; color: {color}; font-weight: bold; font-size: 1.1rem;">
                {row['Correlation']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)

with feature_tab3:
    st.markdown("### Feature Impact on Match Outcomes")
    
    # Box plots for key features by outcome
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(
            df_filtered,
            x='FTResult',
            y='EloDifference',
            color='FTResult',
            title="Elo Difference by Match Outcome",
            labels={'FTResult': 'Outcome', 'EloDifference': 'Elo Difference'},
            color_discrete_map={'H': '#2ecc71', 'D': '#95a5a6', 'A': '#e74c3c'},
            category_orders={'FTResult': ['H', 'D', 'A']}
        )
        fig.update_xaxes(ticktext=['Home Win', 'Draw', 'Away Win'], tickvals=['H', 'D', 'A'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            df_filtered,
            x='FTResult',
            y='Form5Difference',
            color='FTResult',
            title="Form Difference by Match Outcome",
            labels={'FTResult': 'Outcome', 'Form5Difference': 'Form Difference'},
            color_discrete_map={'H': '#2ecc71', 'D': '#95a5a6', 'A': '#e74c3c'},
            category_orders={'FTResult': ['H', 'D', 'A']}
        )
        fig.update_xaxes(ticktext=['Home Win', 'Draw', 'Away Win'], tickvals=['H', 'D', 'A'])
        st.plotly_chart(fig, use_container_width=True)

# Interactive Data Table
st.markdown("---")
st.markdown("## 📋 Explore Match Data")

st.markdown("### 🔎 Sample Matches from Filtered Dataset")

# Allow user to select columns to display
all_columns = df_filtered.columns.tolist()
default_columns = ['MatchDate', 'LeagueName', 'HomeTeam', 'AwayTeam', 'FTResult', 'EloDifference', 'Form5Difference', 'Last3H2H', 'OddHome', 'OddDraw', 'OddAway']
available_defaults = [col for col in default_columns if col in all_columns]

selected_columns = st.multiselect(
    "Select columns to display:",
    options=all_columns,
    default=available_defaults,
    help="Choose which features to display in the data table"
)

# Number of rows to display
n_rows = st.slider("Number of rows to display:", min_value=10, max_value=100, value=20, step=10)

# Sorting option
sort_column = st.selectbox("Sort by:", options=selected_columns if selected_columns else all_columns)
sort_order = st.radio("Sort order:", options=["Descending", "Ascending"], horizontal=True)

# Apply sorting
df_display = df_filtered[selected_columns if selected_columns else all_columns].copy()
df_display = df_display.sort_values(by=sort_column, ascending=(sort_order == "Ascending")).head(n_rows)

# Add outcome labels for readability
if 'FTResult' in df_display.columns:
    df_display['FTResult'] = df_display['FTResult'].map(outcome_map_reverse)

# Display table with styling
st.dataframe(
    df_display,
    use_container_width=True,
    height=400,
    hide_index=True
)

# Export options
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    csv_export = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Displayed Data",
        data=csv_export,
        file_name=f"football_matches_{len(df_display)}_rows.csv",
        mime="text/csv"
    )

with col2:
    st.metric("Rows Shown", f"{len(df_display):,}")

with col3:
    st.info("💡 Tip: Use filters in the sidebar to refine the dataset before exploring")

# Betting Odds Analysis
st.markdown("---")
st.markdown("## 💰 Betting Odds Analysis")

col1, col2 = st.columns(2)

with col1:
    # Implied probabilities
    df_filtered['ImpliedProbHome'] = 1 / df_filtered['OddHome']
    df_filtered['ImpliedProbDraw'] = 1 / df_filtered['OddDraw']
    df_filtered['ImpliedProbAway'] = 1 / df_filtered['OddAway']
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df_filtered['ImpliedProbHome'],
        name='Home Win',
        marker_color='#2ecc71',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig.add_trace(go.Histogram(
        x=df_filtered['ImpliedProbDraw'],
        name='Draw',
        marker_color='#95a5a6',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig.add_trace(go.Histogram(
        x=df_filtered['ImpliedProbAway'],
        name='Away Win',
        marker_color='#e74c3c',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig.update_layout(
        title="Implied Probabilities from Betting Odds",
        xaxis_title="Implied Probability",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Average odds by outcome
    avg_odds = pd.DataFrame({
        'Outcome': ['Home Win', 'Draw', 'Away Win'],
        'Average Odds': [
            df_filtered['OddHome'].mean(),
            df_filtered['OddDraw'].mean(),
            df_filtered['OddAway'].mean()
        ],
        'Implied Probability': [
            df_filtered['ImpliedProbHome'].mean() * 100,
            df_filtered['ImpliedProbDraw'].mean() * 100,
            df_filtered['ImpliedProbAway'].mean() * 100
        ]
    })
    
    fig = px.bar(
        avg_odds,
        x='Outcome',
        y=['Average Odds', 'Implied Probability'],
        title="Average Betting Odds & Implied Probabilities",
        barmode='group',
        height=400,
        color_discrete_sequence=['#1f77b4', '#ff7f0e']
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Data Quality Summary
st.markdown("---")
st.markdown("## ✅ Data Quality Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Completeness")
    completeness = (1 - df_filtered.isnull().sum() / len(df_filtered)) * 100
    
    fig = px.bar(
        x=completeness.values,
        y=completeness.index,
        orientation='h',
        title="Feature Completeness (%)",
        labels={'x': 'Completeness (%)', 'y': 'Feature'},
        color=completeness.values,
        color_continuous_scale='Greens'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Outliers")
    
    outlier_counts = {}
    for col in ['EloDifference', 'Form5Difference', 'GF5Difference']:
        Q1 = df_filtered[col].quantile(0.25)
        Q3 = df_filtered[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df_filtered[col] < (Q1 - 1.5 * IQR)) | (df_filtered[col] > (Q3 + 1.5 * IQR))).sum()
        outlier_counts[col] = outliers
    
    fig = px.bar(
        x=list(outlier_counts.values()),
        y=list(outlier_counts.keys()),
        orientation='h',
        title="Outlier Count by Feature",
        labels={'x': 'Number of Outliers', 'y': 'Feature'},
        color=list(outlier_counts.values()),
        color_continuous_scale='Reds'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.markdown("### Statistics")
    
    st.markdown(f"""
    <div class="metric-card">
        <h4>Dataset Statistics</h4>
        <p><strong>Total Matches:</strong> {len(df_filtered):,}</p>
        <p><strong>Leagues:</strong> {df_filtered['LeagueName'].nunique()}</p>
        <p><strong>Seasons:</strong> {df_filtered['Season'].nunique()}</p>
        <p><strong>Date Range:</strong> {df_filtered['Season'].min()} - {df_filtered['Season'].max()}</p>
        <hr>
        <p><strong>Missing Values:</strong> {df_filtered.isnull().sum().sum()}</p>
        <p><strong>Duplicates:</strong> {df_filtered.duplicated().sum()}</p>
    </div>
    """, unsafe_allow_html=True)

# Download filtered data
st.markdown("---")
st.markdown("## 💾 Export Data")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="filtered_match_data.csv",
        mime="text/csv"
    )

with col2:
    st.metric("Filtered Rows", f"{len(df_filtered):,}")

with col3:
    st.info("💡 Use filters in the sidebar to customize your data view before downloading")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><em>Data Exploration | Football Outcome Predictor</em></p>
</div>
""", unsafe_allow_html=True)
