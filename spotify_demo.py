import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Dark mode Spotify theme
st.set_page_config(
    page_title="Spotify Analytics Lab",
    page_icon="🎶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Spotify dark theme CSS
st.markdown("""
<style>
    /* Spotify dark theme */
    .stApp {
        background: #191414;
        color: #FFFFFF;
    }
    
    /* Custom containers */
    .spotify-card {
        background: #212121;
        border-radius: 12px;
        padding: 25px;
        margin: 15px 0;
        border-left: 4px solid #1DB954;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* Metrics in Spotify style */
    .spotify-metric {
        background: #121212;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #535353;
    }
    
    /* Headers with Spotify green */
    h1, h2, h3 {
        color: #1DB954;
        font-weight: 700;
    }
    
    /* Sliders styled */
    .stSlider > div > div > div {
        background: #1DB954 !important;
    }
    
    /* Tabs dark theme */
    .stTabs [data-baseweb="tab-list"] {
        background: #212121;
        border-radius: 8px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #B3B3B3;
        padding: 10px 24px;
    }
    
    .stTabs [aria-selected="true"] {
        background: #1DB954;
        color: white !important;
        border-radius: 6px;
    }
    
    /* Dataframe styling */
    .dataframe {
        background: #212121 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# HEADER
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown("# 🎵 Spotify Analytics Lab")
    st.markdown("### Linear Algebra Music Analysis")
with col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg", width=80)
with col3:
    st.markdown("### AMTH 222")
    st.markdown("*Hussein Zindonda*")

# Dataset
data = {
    'Tempo': [90, 140, 130, 95, 145, 115, 171, 120, 110, 135, 75, 120, 154, 140, 130, 100, 90, 78, 117, 150],
    'Dance': [75, 60, 70, 65, 68, 72, 80, 55, 58, 62, 78, 72, 65, 70, 68, 55, 82, 75, 85, 73],
    'Acoustic': [25, 15, 10, 30, 20, 8, 5, 60, 45, 18, 2, 35, 8, 12, 15, 40, 25, 10, 5, 18],
    'Streams': [2.8, 2.1, 1.8, 1.5, 2.3, 1.7, 3.5, 1.4, 1.9, 1.2, 1.8, 2.4, 1.9, 1.5, 1.6, 1.3, 2.9, 2.5, 2.1, 1.7]
}
df = pd.DataFrame(data)

# MAIN CONTENT
st.markdown('<div class="spotify-card">', unsafe_allow_html=True)

# SONG BUILDER
st.markdown("### 🎛️ Create Your Hit Song")

col1, col2, col3 = st.columns(3)

with col1:
    tempo = st.slider("**Tempo** (BPM)", 60, 200, 120, 5,
                     help="Beats per minute - higher = faster")
    st.progress((tempo-60)/140)
    
    dance = st.slider("**Danceability**", 0, 100, 70, 5,
                     help="How danceable is the track?")
    st.progress(dance/100)

with col2:
    acoustic = st.slider("**Acousticness**", 0, 100, 20, 5,
                        help="Acoustic vs electronic")
    st.progress(acoustic/100)
    
    energy = st.slider("**Energy**", 0, 100, 80, 5,
                      help="Perceived intensity")
    st.progress(energy/100)

with col3:
    valence = st.slider("**Valence**", 0, 100, 60, 5,
                       help="Musical positivity")
    st.progress(valence/100)
    
    duration = st.slider("**Duration** (sec)", 60, 360, 200, 10,
                        help="Song length")
    st.progress((duration-60)/300)

st.markdown('</div>', unsafe_allow_html=True)

# PREDICTION SECTION
st.markdown('<div class="spotify-card">', unsafe_allow_html=True)
st.markdown("### 📊 Prediction Results")

# Model
X = df[['Tempo', 'Dance', 'Acoustic']]
y = df['Streams'] * 1e9
model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[tempo, dance, acoustic]])[0]
avg_streams = df['Streams'].mean() * 1e9
difference = ((prediction - avg_streams) / avg_streams * 100)

# Metrics in grid
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="spotify-metric">', unsafe_allow_html=True)
    st.markdown("##### Predicted Streams")
    st.markdown(f"## {prediction:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="spotify-metric">', unsafe_allow_html=True)
    st.markdown("##### Performance")
    color = "#1DB954" if difference >= 0 else "#E22134"
    st.markdown(f'<h2 style="color: {color}">{difference:+.1f}%</h2>', unsafe_allow_html=True)
    st.markdown("vs Average")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="spotify-metric">', unsafe_allow_html=True)
    st.markdown("##### Popularity Score")
    popularity = max(0, min(100, 50 + (difference / 2)))
    st.markdown(f"## {popularity:.0f}")
    st.markdown("/100")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="spotify-metric">', unsafe_allow_html=True)
    st.markdown("##### Model Accuracy")
    r2 = model.score(X, y)
    st.markdown(f"## {r2:.3f}")
    st.markdown("R² Score")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# VISUALIZATION
st.markdown('<div class="spotify-card">', unsafe_allow_html=True)
st.markdown("### 📈 Data Visualization")

# Correlation heatmap with dark theme
corr = df.corr()
fig = px.imshow(corr, 
               text_auto='.2f',
               color_continuous_scale='viridis',
               template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# 3D Scatter plot
fig3d = px.scatter_3d(df, 
                     x='Tempo', 
                     y='Dance', 
                     z='Acoustic',
                     size='Streams',
                     color='Streams',
                     hover_name=df.index,
                     template="plotly_dark",
                     title="3D Feature Space")
st.plotly_chart(fig3d, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# MATHEMATICAL DETAILS
with st.expander("🧮 Show Mathematical Foundation", icon="🔢"):
    st.markdown("### Linear Algebra Implementation")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Design Matrix:**")
        st.latex(r'''
        X = \begin{bmatrix}
        x_{11} & x_{12} & x_{13} \\
        x_{21} & x_{22} & x_{23} \\
        \vdots & \vdots & \vdots \\
        x_{20,1} & x_{20,2} & x_{20,3}
        \end{bmatrix}
        ''')
    
    with col2:
        st.markdown("**Normal Equation:**")
        st.latex(r'''
        \beta = (X^T X)^{-1} X^T y
        ''')
    
    st.markdown("**Current Model Coefficients:**")
    coeff_df = pd.DataFrame({
        'Feature': ['Tempo', 'Danceability', 'Acousticness'],
        'Coefficient': model.coef_,
        'Effect': ['Positive' if c > 0 else 'Negative' for c in model.coef_]
    })
    st.dataframe(coeff_df)

# FOOTER
st.markdown("""
<div style="text-align: center; padding: 30px; border-top: 1px solid #535353; margin-top: 40px;">
    <h4 style="color: #1DB954;">🎵 Spotify Analytics Lab</h4>
    <p style="color: #B3B3B3;">AMTH 222 Linear Algebra Project | Multiple Regression Analysis</p>
</div>
""", unsafe_allow_html=True)
