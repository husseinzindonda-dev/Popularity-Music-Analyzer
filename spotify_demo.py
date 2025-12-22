# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ====================
# PAGE CONFIGURATION
# ====================
st.set_page_config(
    page_title="Spotify Hit Predictor Pro",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================
# CUSTOM CSS FOR BETTER UI
# ====================
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Metric cards styling */
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1DB954;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #1DB954;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #191414;
    }
    
    /* Dataframe styling */
    .dataframe {
        border: none !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1DB954;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    
    /* Section dividers */
    .section-divider {
        border-top: 3px solid #1DB954;
        margin: 2rem 0;
        padding-top: 1rem;
    }
    
    /* Info box styling */
    .info-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 10px 0;
    }
    
    /* Custom metric styling */
    .custom-metric {
        background: linear-gradient(135deg, #1DB954, #1ED760);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ====================
# SIDEBAR
# ====================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg", width=100)
    st.markdown("<h1 style='text-align: center; color: #1DB954;'>Spotify Hit Predictor</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 👤 Student Info")
    st.markdown("**Name:** Hussein Zindonda")
    st.markdown("**Course:** AMTH 222")
    st.markdown("**Instructor:** Tamunonye Cheetham-West")
    st.markdown("**Date:** December 2025")
    
    st.markdown("---")
    
    st.markdown("### 📊 Project Info")
    st.markdown("**Dataset:** Top 20 Spotify Songs")
    st.markdown("**Features Analyzed:** 11")
    st.markdown("**Methods:** Linear Regression, PCA, Eigen Analysis")
    
    st.markdown("---")
    
    st.markdown("### 🎯 Quick Start")
    st.markdown("1. Adjust sliders to build your song")
    st.markdown("2. View real-time predictions")
    st.markdown("3. Explore feature correlations")
    st.markdown("4. Check model performance")

# ====================
# MAIN HEADER
# ====================
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown("<h1 style='color: #1DB954; margin-bottom: 0;'>🎵 Spotify Hit Predictor Pro</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #666; margin-top: 0;'>Multi-Feature Linear Algebra Analysis</h3>", unsafe_allow_html=True)
with col_head2:
    st.markdown("<div class='custom-metric'><h4>AMTH 222</h4><p>Linear Algebra Project</p></div>", unsafe_allow_html=True)

st.markdown("---")

# ====================
# INTRODUCTION
# ====================
st.markdown("""
<div class='info-box'>
<h3>🎯 Project Overview</h3>
<p>This interactive tool uses linear algebra to analyze Spotify's top-streamed songs. 
Adjust song features in real-time and see how they affect predicted streams using 
multiple linear regression and principal component analysis.</p>
</div>
""", unsafe_allow_html=True)

# ====================
# DATASET
# ====================
@st.cache_data
def load_data():
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
        'Streams per Second': [1.77e7, 8.79e6, 9.89e6, 5.14e6, 1.06e7, 6.99e6, 1.75e7, 1.18e7, 1.14e7, 7.27e6, 1.02e7, 1.12e7, 6.09e6, 1.03e7, 7.27e6, 5.83e6, 1.22e7, 1.26e7, 1.07e7, 7.98e6],
        'Energy per Loudness': [15.0, 14.17, 20.0, 10.0, 30.0, 17.6, 21.25, 8.13, 11.14, 13.67, 30.67, 11.33, 23.75, 17.4, 16.6, 10.29, 10.0, 20.0, 17.8, 19.25]
    }
    return pd.DataFrame(data)

df = load_data()

# ====================
# INTERACTIVE SONG BUILDER (SECTION 1)
# ====================
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("<h2>🎛️ Interactive Song Builder</h2>", unsafe_allow_html=True)
st.markdown("Adjust the sliders below to create your perfect hit song and see real-time predictions.")

# Create tabs for different feature groups
tab1, tab2 = st.tabs(["🎵 Basic Features", "🎛️ Advanced Features"])

with tab1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎵 Rhythm")
        tempo = st.slider("Tempo (BPM)", 60, 200, 120, 1,
                         help="Beats per minute - typical range: 60-200")
        duration = st.slider("Duration (seconds)", 60, 360, 200, 1,
                           help="Song length in seconds")
    
    with col2:
        st.markdown("### 💃 Feel")
        danceability = st.slider("Danceability (0-100)", 0, 100, 70, 1,
                               help="How suitable for dancing: 0=not danceable, 100=very danceable")
        valence = st.slider("Valence (0-100)", 0, 100, 60, 1,
                          help="Musical positivity: 0=sad/depressing, 100=happy/cheerful")
    
    with col3:
        st.markdown("### 🔊 Sound")
        acousticness = st.slider("Acousticness (0-100)", 0, 100, 20, 1,
                               help="Acoustic vs electronic: 0=electronic, 100=acoustic")
        loudness = st.slider("Loudness (dB)", -15, 0, -5, 1,
                           help="Overall loudness: -15=quiet, 0=loud")

with tab2:
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("### ⚡ Energy")
        energy = st.slider("Energy (0-100)", 0, 100, 80, 1,
                         help="Perceived intensity: 0=calm, 100=intense")
    
    with col5:
        st.markdown("### 📊 Derived Metrics")
        # Calculate derived features
        streams_per_second = ((tempo * danceability) / (duration * (acousticness + 1))) * 1000
        energy_per_loudness = energy / abs(loudness) if loudness != 0 else 0
        
        st.metric("Streams/Second", f"{streams_per_second:,.0f}")
        st.metric("Energy/Loudness", f"{energy_per_loudness:.1f}")
    
    with col6:
        st.markdown("### 🎯 Current Values")
        st.write(f"**Tempo:** {tempo} BPM")
        st.write(f"**Danceability:** {danceability}/100")
        st.write(f"**Acousticness:** {acousticness}/100")

# ====================
# PREDICTION SECTION (SECTION 2)
# ====================
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# Prepare data for model
X_features = ['Tempo (BPM)', 'Danceability (/100)', 'Acousticness (/100)', 
              'Loudness (dB)', 'Energy (/100)', 'Valence (/100)', 
              'Duration (seconds)']

X = df[X_features]
y = df['Streams (billions)'] * 1e9

# Fit model
@st.cache_resource
def train_model():
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model()

# Prediction function
def predict_all_features(tempo, dance, acoustic, loud, energy_val, valence_val, duration_sec):
    features = np.array([[tempo, dance, acoustic, loud, energy_val, valence_val, duration_sec]])
    return model.predict(features)[0]

# Make prediction
prediction = predict_all_features(tempo, danceability, acousticness, 
                                  loudness, energy, valence, duration)

# Display prediction cards
st.markdown("<h2>📊 Stream Prediction Results</h2>", unsafe_allow_html=True)

col_pred1, col_pred2, col_pred3, col_pred4 = st.columns(4)

avg_streams = df['Streams (billions)'].mean() * 1e9
difference = ((prediction - avg_streams) / avg_streams * 100)
estimated_popularity = max(0, min(100, 50 + (difference / 2)))

with col_pred1:
    st.metric(
        label="🎯 Predicted Streams",
        value=f"{prediction:,.0f}",
        delta=None
    )

with col_pred2:
    delta_color = "normal" if difference >= 0 else "inverse"
    st.metric(
        label="📈 Vs. Average",
        value=f"{difference:+.1f}%",
        delta=None,
        delta_color=delta_color
    )

with col_pred3:
    st.metric(
        label="⭐ Estimated Popularity",
        value=f"{estimated_popularity:.0f}/100",
        delta=None
    )

with col_pred4:
    r_squared = model.score(X, y)
    st.metric(
        label="📊 Model R² Score",
        value=f"{r_squared:.3f}",
        delta=None
    )

# Performance summary
if difference > 10:
    st.success(f"🎉 **Excellent!** Your song would perform **{difference:+.1f}% better** than average hits!")
elif difference > 0:
    st.info(f"📈 **Good!** Your song would perform **{difference:+.1f}% better** than average.")
elif difference > -10:
    st.warning(f"⚠️ **Average.** Your song would perform **{difference:+.1f}% worse** than average.")
else:
    st.error(f"❌ **Needs work.** Your song would perform **{difference:+.1f}% worse** than average.")

# ====================
# FEATURE ANALYSIS (SECTION 3)
# ====================
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("<h2>📈 Feature Correlation Analysis</h2>", unsafe_allow_html=True)

# Correlation matrix
numerical_features = ['Tempo (BPM)', 'Popularity (/100)', 'Streams (billions)', 
                      'Danceability (/100)', 'Acousticness (/100)', 'Loudness (dB)',
                      'Duration (seconds)', 'Energy (/100)', 'Valence (/100)',
                      'Streams per Second', 'Energy per Loudness']

corr_matrix = df[numerical_features].corr()

# Interactive correlation matrix
fig_corr = px.imshow(corr_matrix, 
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    aspect="auto",
                    title="Spotify Features Correlation Matrix",
                    labels=dict(color="Correlation"))
fig_corr.update_layout(height=600)
st.plotly_chart(fig_corr, use_container_width=True)

# Key insights in expandable sections
with st.expander("🔍 View Key Correlation Insights"):
    col_ins1, col_ins2 = st.columns(2)
    
    with col_ins1:
        st.markdown("### ✅ Strong Positive Correlations")
        st.markdown("- **Popularity ↔ Streams:** +0.97")
        st.markdown("- **Loudness ↔ Energy/Loudness:** +0.94")
        st.markdown("- **Danceability ↔ Valence:** +0.71")
        st.markdown("- **Streams/Second ↔ Popularity:** +0.74")
    
    with col_ins2:
        st.markdown("### ⚠️ Strong Negative Correlations")
        st.markdown("- **Acousticness ↔ Loudness:** -0.81")
        st.markdown("- **Acousticness ↔ Energy:** -0.74")
        st.markdown("- **Acousticness ↔ Danceability:** -0.64")
        st.markdown("- **Duration ↔ Streams/Second:** -0.53")

# ====================
# REGRESSION MODEL (SECTION 4)
# ====================
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("<h2>🧮 Linear Regression Model</h2>", unsafe_allow_html=True)

# Display regression equation
st.markdown("### Regression Equation:")
st.latex(r'''
\text{Predicted Streams} = \beta_0 + \beta_1 \times \text{Tempo} + \beta_2 \times \text{Danceability} + \cdots
''')

# Display coefficients
coeff_df = pd.DataFrame({
    'Feature': X_features,
    'Coefficient': model.coef_,
    'Impact': ['Positive' if c > 0 else 'Negative' for c in model.coef_]
}).round(2)

st.dataframe(coeff_df.style.format({'Coefficient': '{:,.0f}'})
             .background_gradient(subset=['Coefficient'], cmap='RdYlGn')
             .set_properties(**{'text-align': 'left'}))

# ====================
# FEATURE IMPORTANCE (SECTION 5)
# ====================
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("<h2>📊 Feature Impact Analysis</h2>", unsafe_allow_html=True)

# Feature importance chart
feature_importance = pd.DataFrame({
    'Feature': X_features,
    'Impact Score': abs(model.coef_) / max(abs(model.coef_)) * 100
}).sort_values('Impact Score', ascending=True)

fig_importance = px.bar(feature_importance, 
                       x='Impact Score', 
                       y='Feature',
                       orientation='h',
                       title="Relative Impact of Features on Stream Prediction",
                       color='Impact Score',
                       color_continuous_scale='viridis',
                       text='Impact Score')
fig_importance.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig_importance.update_layout(height=400)
st.plotly_chart(fig_importance, use_container_width=True)

# ====================
# WHAT-IF ANALYSIS (SECTION 6)
# ====================
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("<h2>🔬 What-If Scenarios</h2>", unsafe_allow_html=True)

what_if_col1, what_if_col2, what_if_col3 = st.columns(3)

with what_if_col1:
    with st.container():
        st.markdown("#### More Danceable")
        new_dance = min(100, danceability + 20)
        pred_more_dance = predict_all_features(tempo, new_dance, acousticness, 
                                              loudness, energy, valence, duration)
        dance_impact = ((pred_more_dance - prediction) / prediction * 100)
        st.metric("+20 Danceability", f"{dance_impact:+.1f}%", 
                 help="Effect of increasing danceability by 20 points")

with what_if_col2:
    with st.container():
        st.markdown("#### Less Acoustic")
        new_acoustic = max(0, acousticness - 30)
        pred_less_acoustic = predict_all_features(tempo, danceability, new_acoustic,
                                                 loudness, energy, valence, duration)
        acoustic_impact = ((pred_less_acoustic - prediction) / prediction * 100)
        st.metric("-30 Acousticness", f"{acoustic_impact:+.1f}%",
                 help="Effect of reducing acousticness by 30 points")

with what_if_col3:
    with st.container():
        st.markdown("#### Higher Energy")
        new_energy = min(100, energy + 20)
        pred_more_energy = predict_all_features(tempo, danceability, acousticness,
                                               loudness, new_energy, valence, duration)
        energy_impact = ((pred_more_energy - prediction) / prediction * 100)
        st.metric("+20 Energy", f"{energy_impact:+.1f}%",
                 help="Effect of increasing energy by 20 points")

# ====================
# LINEAR ALGEBRA INSIGHTS (SECTION 7)
# ====================
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("<h2>🧮 Linear Algebra Foundation</h2>", unsafe_allow_html=True)

# Mathematical foundation in tabs
tab_math1, tab_math2, tab_math3 = st.tabs(["Matrix Formulation", "Normal Equations", "Eigen Analysis"])

with tab_math1:
    st.markdown("### Design Matrix X (20 songs × 7 features):")
    st.latex(r'''
    X = \begin{bmatrix}
    \text{Tempo}_1 & \text{Danceability}_1 & \cdots & \text{Duration}_1 \\
    \text{Tempo}_2 & \text{Danceability}_2 & \cdots & \text{Duration}_2 \\
    \vdots & \vdots & \ddots & \vdots \\
    \text{Tempo}_{20} & \text{Danceability}_{20} & \cdots & \text{Duration}_{20}
    \end{bmatrix}
    ''')
    
    st.markdown("### Target Vector y (Stream counts):")
    st.latex(r'''
    y = \begin{bmatrix}
    \text{Streams}_1 \\
    \text{Streams}_2 \\
    \vdots \\
    \text{Streams}_{20}
    \end{bmatrix}
    ''')

with tab_math2:
    st.markdown("### Least Squares Solution:")
    st.latex(r'''
    \min_{\beta} \|X\beta - y\|^2
    ''')
    
    st.markdown("### Normal Equation:")
    st.latex(r'''
    \beta = (X^T X)^{-1} X^T y
    ''')
    
    st.markdown("### Prediction:")
    st.latex(r'''
    \hat{y} = X\beta
    ''')

with tab_math3:
    # Calculate PCA
    X_scaled = StandardScaler().fit_transform(X)
    cov_matrix = np.cov(X_scaled.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    st.markdown("### Covariance Matrix:")
    st.latex(r'''
    C = X^T X
    ''')
    
    st.markdown("### Eigenvalue Decomposition:")
    st.latex(r'''
    C v = \lambda v
    ''')
    
    st.markdown(f"### Explained Variance:")
    st.markdown(f"- **PC1:** {eigenvalues[0]/sum(eigenvalues)*100:.1f}%")
    st.markdown(f"- **PC2:** {eigenvalues[1]/sum(eigenvalues)*100:.1f}%")
    st.markdown(f"- **PC3:** {eigenvalues[2]/sum(eigenvalues)*100:.1f}%")
    
    # Display eigenvectors
    with st.expander("View Principal Components"):
        col_pc1, col_pc2 = st.columns(2)
        with col_pc1:
            st.markdown("**PC1 Loadings:**")
            for i, feature in enumerate(X_features):
                st.write(f"{feature}: {eigenvectors[i, 0]:.3f}")
        with col_pc2:
            st.markdown("**PC2 Loadings:**")
            for i, feature in enumerate(X_features):
                st.write(f"{feature}: {eigenvectors[i, 1]:.3f}")

# ====================
# FOOTER
# ====================
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;'>
    <h3 style='color: #1DB954;'>AMTH 222 - Linear Algebra Project</h3>
    <p><strong>Student:</strong> Hussein Zindonda | <strong>Instructor:</strong> Tamunonye Cheetham-West</p>
    <p><strong>Methods:</strong> Multiple Linear Regression • Principal Component Analysis • Eigen Decomposition</p>
    <p><strong>Dataset:</strong> Top 20 Spotify Songs with 11 Audio Features</p>
</div>
""", unsafe_allow_html=True)
