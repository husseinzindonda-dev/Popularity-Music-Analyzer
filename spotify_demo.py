import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(
    page_title="Spotify Analytics Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern look
st.markdown("""
<style>
    /* Modern gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main content card */
    .main-card {
        background: white;
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    /* Feature card styling */
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 5px solid #1DB954;
    }
    
    /* Modern metric cards */
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Custom slider styling */
    .stSlider > div > div > div {
        background: #1DB954;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        border-radius: 10px 10px 0 0;
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(45deg, #1DB954, #1ED760);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
</style>
""", unsafe_allow_html=True)

# HEADER
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("# 🎵 Spotify Hit Predictor")
    st.markdown("### AI-Powered Music Analytics Dashboard")
with col2:
    st.markdown("### AMTH 222")
    st.markdown("*Linear Algebra Project*")

# Main container
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    # Dataset (same as before)
    data = {
        'Artist': ['Post Malone', 'Juice WRLD', 'Lil Uzi Vert', 'J. Cole', 'Post Malone', 
                  'Travis Scott', 'The Weeknd', 'XXXTENTACION', 'XXXTENTACION', 'Juice WRLD',
                  'Kendrick Lamar', 'Post Malone', 'Travis Scott', 'Lil Baby', 'Post Malone',
                  'Post Malone', 'Glass Animals', 'Drake', 'Roddy Ricch', 'J. Cole'],
        'Track Name': ['Sunflower', 'Lucid Dreams', 'XO TOUR Llif3', 'No Role Modelz', 'rockstar',
                      'goosebumps', 'Blinding Lights', 'Jocelyn Flores', 'SAD!', 'All Girls Are The Same',
                      'HUMBLE.', 'Circles', 'SICKO MODE', 'Drip Too Hard', 'Congratulations',
                      'I Fall Apart', 'Heat Waves', 'God\'s Plan', 'The Box', 'MIDDLE CHILD'],
        'Tempo (BPM)': [90, 140, 130, 95, 145, 115, 171, 120, 110, 135, 75, 120, 154, 140, 130, 100, 90, 78, 117, 150],
        'Popularity (/100)': [95, 92, 90, 88, 93, 89, 98, 87, 91, 86, 92, 94, 91, 88, 89, 87, 96, 95, 93, 90],
        'Streams (billions)': [2.8, 2.1, 1.8, 1.5, 2.3, 1.7, 3.5, 1.4, 1.9, 1.2, 1.8, 2.4, 1.9, 1.5, 1.6, 1.3, 2.9, 2.5, 2.1, 1.7],
        'Danceability (/100)': [75, 60, 70, 65, 68, 72, 80, 55, 58, 62, 78, 72, 65, 70, 68, 55, 82, 75, 85, 73],
        'Acousticness (/100)': [25, 15, 10, 30, 20, 8, 5, 60, 45, 18, 2, 35, 8, 12, 15, 40, 25, 10, 5, 18],
        'Loudness (dB)': [-5, -6, -4, -7, -3, -5, -4, -8, -7, -6, -3, -6, -4, -5, -5, -7, -6, -4, -5, -4],
        'Duration (seconds)': [158, 239, 182, 292, 218, 243, 200, 119, 166, 165, 177, 215, 312, 145, 220, 223, 238, 199, 196, 213],
        'Energy (/100)': [75, 85, 80, 70, 90, 88, 85, 65, 78, 82, 92, 68, 95, 87, 83, 72, 60, 80, 89, 77],
        'Valence (/100)': [65, 45, 55, 70, 60, 58, 40, 25, 35, 48, 68, 72, 62, 75, 52, 38, 80, 85, 78, 65],
    }
    df = pd.DataFrame(data)
    
    # ===== SONG BUILDER =====
    st.markdown("## 🎛️ Song Feature Builder")
    
    # Feature cards in 4 columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🎵 Tempo")
        tempo = st.slider("BPM", 60, 200, 120, 1, label_visibility="collapsed")
        st.metric("Value", f"{tempo} BPM")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### ⏱️ Duration")
        duration = st.slider("Seconds", 60, 360, 200, 1, label_visibility="collapsed")
        st.metric("Value", f"{duration}s")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 💃 Danceability")
        danceability = st.slider("Score", 0, 100, 70, 1, label_visibility="collapsed")
        st.progress(danceability/100)
        st.metric("Value", f"{danceability}/100")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 😊 Valence")
        valence = st.slider("Positivity", 0, 100, 60, 1, label_visibility="collapsed")
        st.progress(valence/100)
        st.metric("Value", f"{valence}/100")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🎸 Acousticness")
        acousticness = st.slider("Acoustic", 0, 100, 20, 1, label_visibility="collapsed")
        st.progress(acousticness/100)
        st.metric("Value", f"{acousticness}/100")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🔊 Loudness")
        loudness = st.slider("dB", -15, 0, -5, 1, label_visibility="collapsed")
        st.metric("Value", f"{loudness} dB")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### ⚡ Energy")
        energy = st.slider("Energy", 0, 100, 80, 1, label_visibility="collapsed")
        st.progress(energy/100)
        st.metric("Value", f"{energy}/100")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== PREDICTION RESULTS =====
    st.markdown("## 📊 Prediction Results")
    
    # Train model
    X = df[['Tempo (BPM)', 'Danceability (/100)', 'Acousticness (/100)', 
            'Loudness (dB)', 'Energy (/100)', 'Valence (/100)', 'Duration (seconds)']]
    y = df['Streams (billions)'] * 1e9
    
    model = LinearRegression()
    model.fit(X, y)
    
    def predict_streams(features):
        return model.predict([features])[0]
    
    features = [tempo, danceability, acousticness, loudness, energy, valence, duration]
    prediction = predict_streams(features)
    avg_streams = df['Streams (billions)'].mean() * 1e9
    difference = ((prediction - avg_streams) / avg_streams * 100)
    
    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Predicted")
        st.markdown(f"# {prediction:,.0f}")
        st.markdown("Streams")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Vs Average")
        color = "green" if difference >= 0 else "red"
        st.markdown(f'<h1 style="color: {color}">{difference:+.1f}%</h1>', unsafe_allow_html=True)
        st.markdown("Performance")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### ⭐ Popularity")
        popularity = max(0, min(100, 50 + (difference / 2)))
        st.markdown(f"# {popularity:.0f}")
        st.markdown("/100 Score")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Model Fit")
        r2 = model.score(X, y)
        st.markdown(f"# {r2:.3f}")
        st.markdown("R² Score")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== VISUALIZATION =====
    st.markdown("## 📈 Feature Analysis")
    
    # Correlation matrix
    corr_matrix = df[['Tempo (BPM)', 'Popularity (/100)', 'Streams (billions)', 
                     'Danceability (/100)', 'Acousticness (/100)', 'Loudness (dB)',
                     'Duration (seconds)', 'Energy (/100)', 'Valence (/100)']].corr()
    
    fig = px.imshow(corr_matrix, 
                   text_auto='.2f',
                   color_continuous_scale='RdBu',
                   aspect="auto",
                   title="Feature Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    # ===== FEATURE IMPORTANCE =====
    st.markdown("## 🎯 Feature Impact")
    
    importance = pd.DataFrame({
        'Feature': ['Tempo', 'Danceability', 'Acousticness', 'Loudness', 'Energy', 'Valence', 'Duration'],
        'Impact': abs(model.coef_) / max(abs(model.coef_)) * 100
    }).sort_values('Impact')
    
    fig2 = px.bar(importance, 
                 x='Impact', 
                 y='Feature',
                 orientation='h',
                 color='Impact',
                 color_continuous_scale='viridis',
                 title="Feature Importance in Stream Prediction")
    st.plotly_chart(fig2, use_container_width=True)
    
    # ===== LINEAR ALGEBRA SECTION =====
    with st.expander("🧮 Show Linear Algebra Details", expanded=False):
        st.markdown("### Mathematical Foundation")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Design Matrix X (20×7):**")
            st.latex(r'''
            X = \begin{bmatrix}
            \text{Features}_1 \\
            \text{Features}_2 \\
            \vdots \\
            \text{Features}_{20}
            \end{bmatrix}
            ''')
        
        with col2:
            st.markdown("**Normal Equation:**")
            st.latex(r'''
            \beta = (X^T X)^{-1} X^T y
            ''')
        
        # PCA
        X_scaled = StandardScaler().fit_transform(X)
        cov_matrix = np.cov(X_scaled.T)
        eigenvalues, _ = np.linalg.eigh(cov_matrix)
        eigenvalues = sorted(eigenvalues, reverse=True)
        
        st.markdown(f"**Principal Component Analysis:**")
        st.markdown(f"- PC1 explains {eigenvalues[0]/sum(eigenvalues)*100:.1f}% of variance")
        st.markdown(f"- PC2 explains {eigenvalues[1]/sum(eigenvalues)*100:.1f}% of variance")
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-card

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: white;">
    <h3>🎵 Spotify Hit Predictor</h3>
    <p><strong>AMTH 222 Linear Algebra Project</strong> | Hussein Zindonda</p>
    <p>Multiple Linear Regression • Principal Component Analysis • Eigen Decomposition</p>
</div>
""", unsafe_allow_html=True)
