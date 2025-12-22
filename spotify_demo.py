import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Clean, minimalist design
st.set_page_config(
    page_title="Spotify Analytics",
    page_icon="📊",
    layout="wide"
)

# Minimal CSS
st.markdown("""
<style>
    .stApp {
        background: #f8f9fa;
    }
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    .section {
        background: white;
        border-radius: 10px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 4px solid #1DB954;
    }
    .metric-box {
        text-align: center;
        padding: 15px;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        background: white;
    }
    h2 {
        color: #1DB954;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-container">
    <h1 style="text-align: center; color: #1DB954;">🎵 Spotify Stream Predictor</h1>
    <p style="text-align: center; color: #666; font-size: 1.1em;">
        Linear Algebra Analysis of Music Features | AMTH 222 Project
    </p>
""", unsafe_allow_html=True)

# Dataset
data = {
    'Tempo': [90, 140, 130, 95, 145, 115, 171, 120, 110, 135, 75, 120, 154, 140, 130, 100, 90, 78, 117, 150],
    'Danceability': [75, 60, 70, 65, 68, 72, 80, 55, 58, 62, 78, 72, 65, 70, 68, 55, 82, 75, 85, 73],
    'Acousticness': [25, 15, 10, 30, 20, 8, 5, 60, 45, 18, 2, 35, 8, 12, 15, 40, 25, 10, 5, 18],
    'Loudness': [-5, -6, -4, -7, -3, -5, -4, -8, -7, -6, -3, -6, -4, -5, -5, -7, -6, -4, -5, -4],
    'Duration': [158, 239, 182, 292, 218, 243, 200, 119, 166, 165, 177, 215, 312, 145, 220, 223, 238, 199, 196, 213],
    'Energy': [75, 85, 80, 70, 90, 88, 85, 65, 78, 82, 92, 68, 95, 87, 83, 72, 60, 80, 89, 77],
    'Valence': [65, 45, 55, 70, 60, 58, 40, 25, 35, 48, 68, 72, 62, 75, 52, 38, 80, 85, 78, 65],
    'Streams': [2.8, 2.1, 1.8, 1.5, 2.3, 1.7, 3.5, 1.4, 1.9, 1.2, 1.8, 2.4, 1.9, 1.5, 1.6, 1.3, 2.9, 2.5, 2.1, 1.7]
}
df = pd.DataFrame(data) * 1e9  # Convert to actual streams

# SECTION 1: Input Controls
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("### 🎛️ Configure Song Features")

cols = st.columns(7)
features = {}
with cols[0]:
    features['tempo'] = st.slider("Tempo", 60, 200, 120)
    st.caption("BPM")
with cols[1]:
    features['danceability'] = st.slider("Dance", 0, 100, 70)
    st.caption("/100")
with cols[2]:
    features['acousticness'] = st.slider("Acoustic", 0, 100, 20)
    st.caption("/100")
with cols[3]:
    features['loudness'] = st.slider("Loudness", -15, 0, -5)
    st.caption("dB")
with cols[4]:
    features['energy'] = st.slider("Energy", 0, 100, 80)
    st.caption("/100")
with cols[5]:
    features['valence'] = st.slider("Valence", 0, 100, 60)
    st.caption("/100")
with cols[6]:
    features['duration'] = st.slider("Duration", 60, 360, 200)
    st.caption("seconds")
st.markdown('</div>', unsafe_allow_html=True)

# SECTION 2: Predictions
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("### 📈 Stream Prediction")

# Model
X = df[['Tempo', 'Danceability', 'Acousticness', 'Loudness', 'Energy', 'Valence', 'Duration']]
y = df['Streams']
model = LinearRegression()
model.fit(X, y)

# Prediction
input_features = [[features['tempo'], features['danceability'], features['acousticness'],
                   features['loudness'], features['energy'], features['valence'], features['duration']]]
prediction = model.predict(input_features)[0]

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Predicted Streams", f"{prediction:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    avg_streams = df['Streams'].mean()
    difference = ((prediction - avg_streams) / avg_streams * 100)
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Vs Average", f"{difference:+.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Model R²", f"{model.score(X, y):.3f}")
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# SECTION 3: Visualization
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("### 📊 Data Analysis")

tab1, tab2 = st.tabs(["Correlations", "Feature Importance"])

with tab1:
    corr = df.corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient')
    
    fig2 = px.bar(importance, x='Coefficient', y='Feature', orientation='h',
                 color='Coefficient', color_continuous_scale='RdYlGn')
    st.plotly_chart(fig2, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# SECTION 4: Math Details
with st.expander("Show Linear Algebra Details"):
    st.markdown("""
    ### Mathematical Model
    Multiple linear regression using the normal equation:
    
    $$
    \\beta = (X^T X)^{-1} X^T y
    $$
    
    Where:
    - $X$ is the 20×7 feature matrix
    - $y$ is the vector of stream counts
    - $\\beta$ contains the regression coefficients
    """)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; color: #666; font-size: 0.9em;">
    <p>AMTH 222 Linear Algebra Project | Hussein Zindonda</p>
</div>
</div>
""", unsafe_allow_html=True)
