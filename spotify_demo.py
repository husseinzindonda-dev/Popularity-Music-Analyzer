# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ====================
# SPOTIFY DARK THEME CONFIG
# ====================
st.set_page_config(
    page_title="Spotify Hit Predictor Pro",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Spotify dark theme CSS
st.markdown("""
<style>
    /* Spotify dark theme background */
    .stApp {
        background: #191414;
        color: #FFFFFF;
    }
    
    /* Main content cards */
    .spotify-card {
        background: #212121;
        border-radius: 12px;
        padding: 25px;
        margin: 15px 0;
        border-left: 4px solid #1DB954;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* Metric cards in Spotify style */
    .spotify-metric {
        background: #121212;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #535353;
        transition: transform 0.3s;
    }
    
    .spotify-metric:hover {
        transform: translateY(-3px);
        border-color: #1DB954;
    }
    
    /* Headers with Spotify green */
    h1, h2, h3 {
        color: #1DB954;
        font-weight: 700;
    }
    
    /* Custom slider styling */
    .stSlider > div > div > div {
        background: #1DB954 !important;
    }
    
    /* Dataframe styling for dark mode */
    .dataframe {
        background: #212121 !important;
        color: white !important;
    }
    
    /* Tables in dark mode */
    table {
        background: #212121 !important;
        color: white !important;
    }
    
    th {
        background: #1DB954 !important;
        color: white !important;
    }
    
    td {
        background: #2a2a2a !important;
        color: white !important;
    }
    
    /* Tabs dark theme */
    .stTabs [data-baseweb="tab-list"] {
        background: #212121;
        border-radius: 8px;
        padding: 5px;
        border: 1px solid #535353;
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
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #212121;
        color: #1DB954;
        border: 1px solid #535353;
    }
    
    /* Info, warning, success boxes */
    .stAlert {
        background: #2a2a2a;
        border: 1px solid #535353;
    }
    
    /* Section dividers */
    .section-divider {
        border-top: 2px solid #535353;
        margin: 2rem 0;
        padding-top: 1rem;
    }
    
    /* Latex styling for dark mode */
    .stLaTeX {
        background: #2a2a2a;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #535353;
    }
</style>
""", unsafe_allow_html=True)

# ====================
# SIDEBAR - SPOTIFY STYLE
# ====================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg", width=80)
    st.markdown("<h2 style='text-align: center; color: #1DB954;'>Spotify Hit Lab</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    st.markdown("### 👤 Student Info")
    st.markdown("**Name:** Hussein Zindonda")
    st.markdown("**Course:** AMTH 222")
    st.markdown("**Instructor:** Tamunonye Cheetham-West")
    st.markdown("**Date:** December 2025")
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    st.markdown("### 📊 Dataset Info")
    st.markdown("**Songs:** 20 Top Hits")
    st.markdown("**Features:** 11 Audio Metrics")
    st.markdown("**Stream Range:** 1.2B - 3.5B")
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    st.markdown("### 🧮 Methods Used")
    st.markdown("- Multiple Linear Regression")
    st.markdown("- Principal Component Analysis")
    st.markdown("- Eigenvector Decomposition")
    st.markdown("- Correlation Analysis")

# ====================
# MAIN CONTENT
# ====================

# HEADER
col_head1, col_head2, col_head3 = st.columns([3, 1, 1])
with col_head1:
    st.markdown("# 🎵 Spotify Hit Predictor Pro")
    st.markdown("### Multi-Feature Linear Algebra Analysis")
with col_head2:
    st.markdown('<div class="spotify-metric">', unsafe_allow_html=True)
    st.markdown("##### AMTH 222")
    st.markdown("**Linear Algebra**")
    st.markdown('</div>', unsafe_allow_html=True)
with col_head3:
    st.markdown('<div class="spotify-metric">', unsafe_allow_html=True)
    st.markdown("##### Dataset")
    st.markdown("**20 Songs**")
    st.markdown("**11 Features**")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# ====================
# COMPREHENSIVE DATASET
# ====================
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

df = pd.DataFrame(data)

# ====================
# 1. CORRELATION ANALYSIS
# ====================
st.markdown('<div class="spotify-card">', unsafe_allow_html=True)
st.markdown("## 📈 1. Comprehensive Correlation Analysis")

# Calculate full correlation matrix
numerical_features = ['Tempo (BPM)', 'Popularity (/100)', 'Streams (billions)', 
                      'Danceability (/100)', 'Acousticness (/100)', 'Loudness (dB)',
                      'Duration (seconds)', 'Energy (/100)', 'Valence (/100)',
                      'Streams per Second', 'Energy per Loudness']

corr_matrix = df[numerical_features].corr()

# Display correlation matrix with dark theme
fig = px.imshow(corr_matrix, 
                text_auto='.2f',
                color_continuous_scale='viridis',
                aspect="auto",
                title="Comprehensive Feature Correlation Matrix")
fig.update_layout(
    plot_bgcolor='#212121',
    paper_bgcolor='#212121',
    font_color='white'
)
st.plotly_chart(fig, use_container_width=True)

# Key insights in columns
col1, col2, col3 = st.columns(3)
with col1:
    st.success("""
    **Strongest Correlations:**
    - Popularity ↔ Streams: +0.97
    - Loudness ↔ Energy/Loudness: +0.94
    - Danceability ↔ Valence: +0.71
    """)

with col2:
    st.warning("""
    **Negative Relationships:**
    - Acousticness ↔ Loudness: -0.81
    - Acousticness ↔ Energy: -0.74
    - Acousticness ↔ Danceability: -0.64
    """)

with col3:
    st.info("""
    **New Insights:**
    - Streams/Second ↔ Popularity: +0.74
    - Duration ↔ Streams/Second: -0.53
    - Energy ↔ Acousticness: -0.74
    """)
st.markdown('</div>', unsafe_allow_html=True)

# ====================
# 2. MULTI-FEATURE LINEAR REGRESSION
# ====================
st.markdown('<div class="spotify-card">', unsafe_allow_html=True)
st.markdown("## 🎯 2. Multi-Feature Stream Prediction Model")

# Prepare data for regression
X_features = ['Tempo (BPM)', 'Danceability (/100)', 'Acousticness (/100)', 
              'Loudness (dB)', 'Energy (/100)', 'Valence (/100)', 
              'Duration (seconds)']

X = df[X_features]
y = df['Streams (billions)'] * 1e9  # Convert to actual streams

# Fit the model
model = LinearRegression()
model.fit(X, y)

st.markdown("### Regression Equation with All Features:")
equation = "Predicted Streams = "
equation += f"{model.intercept_:,.0f} "
for i, feature in enumerate(X_features):
    coef = model.coef_[i]
    sign = "+" if coef >= 0 else "-"
    equation += f"{sign} {abs(coef):,.0f} × {feature} "
st.latex(f"\\text{{{equation}}}")

# Display coefficients with dark theme styling
coeff_df = pd.DataFrame({
    'Feature': X_features,
    'Coefficient': model.coef_,
    'Impact': ['Positive' if c > 0 else 'Negative' for c in model.coef_],
    'Magnitude': [f"{abs(c):,.0f}" for c in model.coef_]
})

# Simple dataframe without background_gradient
st.dataframe(coeff_df.style.format({'Coefficient': '{:,.0f}'}))
st.markdown('</div>', unsafe_allow_html=True)

# ====================
# 3. INTERACTIVE PREDICTOR WITH ALL FEATURES
# ====================
st.markdown('<div class="spotify-card">', unsafe_allow_html=True)
st.markdown("## 🎛️ 3. Interactive Song Builder")

st.markdown("**Adjust all features to build your perfect hit song:**")

# Create two rows of sliders
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 🎵 Rhythm")
    tempo = st.slider("Tempo (BPM)", 60, 200, 120, 1)
    st.metric("Selected", f"{tempo} BPM", delta=None)
    duration = st.slider("Duration (sec)", 60, 360, 200, 1)
    st.metric("Selected", f"{duration}s", delta=None)
    
with col2:
    st.markdown("### 💃 Feel")
    danceability = st.slider("Danceability", 0, 100, 70, 1)
    st.progress(danceability/100)
    valence = st.slider("Valence (Positivity)", 0, 100, 60, 1)
    st.progress(valence/100)
    
with col3:
    st.markdown("### 🔊 Sound")
    acousticness = st.slider("Acousticness", 0, 100, 20, 1)
    st.progress(acousticness/100)
    loudness = st.slider("Loudness (dB)", -15, 0, -5, 1)
    st.metric("Selected", f"{loudness} dB", delta=None)
    
with col4:
    st.markdown("### ⚡ Energy")
    energy = st.slider("Energy", 0, 100, 80, 1)
    st.progress(energy/100)

# Calculate derived features
streams_per_second = ((tempo * danceability) / (duration * (acousticness + 1))) * 1000
energy_per_loudness = energy / abs(loudness) if loudness != 0 else 0

# Display current settings
st.markdown("### Current Song Profile:")
current_profile = pd.DataFrame({
    'Feature': X_features + ['Streams/Second', 'Energy/Loudness'],
    'Value': [tempo, danceability, acousticness, loudness, energy, valence, duration,
             f"{streams_per_second:,.0f}", f"{energy_per_loudness:.1f}"]
})
st.table(current_profile)
st.markdown('</div>', unsafe_allow_html=True)

# ====================
# 4. PREDICTION ENGINE
# ====================
# Replace your entire prediction section with this:

st.markdown('<div class="spotify-card">', unsafe_allow_html=True)
st.markdown("## 📊 4. Stream Prediction")

def predict_all_features(tempo, dance, acoustic, loud, energy_val, valence_val, duration_sec):
    """Predict streams using all features"""
    features = np.array([[tempo, dance, acoustic, loud, energy_val, valence_val, duration_sec]])
    return model.predict(features)[0]

# Calculate prediction
prediction = predict_all_features(tempo, danceability, acousticness, 
                                  loudness, energy, valence, duration)

# Calculate additional metrics
avg_streams = df['Streams (billions)'].mean() * 1e9
difference = ((prediction - avg_streams) / avg_streams * 100)

# Display results in Spotify metric cards
st.markdown("### Prediction Results")

col_pred1, col_pred2, col_pred3 = st.columns(3)

with col_pred1:
    st.markdown('<div class="spotify-metric">', unsafe_allow_html=True)
    
    # Predicted Streams in text bar
    stream_text = f"{prediction:,.0f}"
    st.markdown("##### 🎯 Predicted Streams")
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1DB954 0%, #1ED760 100%);
        color: white;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
        font-size: 1.6em;
        font-weight: bold;
        font-family: 'Courier New', monospace;
        box-shadow: 0 4px 8px rgba(29, 185, 84, 0.3);
        border: 2px solid white;
    ">
    {stream_text}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p style="text-align: center; color: #B3B3B3; margin-top: 5px;">streams</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_pred2:
    st.markdown('<div class="spotify-metric">', unsafe_allow_html=True)
    
    # Vs Average in text bar
    diff_text = f"{difference:+.1f}%"
    diff_color = "#1DB954" if difference >= 0 else "#E22134"
    st.markdown("##### 📈 Vs. Average")
    st.markdown(f"""
    <div style="
        background: {diff_color};
        color: white;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
        font-size: 1.6em;
        font-weight: bold;
        font-family: 'Courier New', monospace;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        border: 2px solid white;
    ">
    {diff_text}
    </div>
    """, unsafe_allow_html=True)
    
    performance = "Better" if difference >= 0 else "Worse"
    st.markdown(f'<p style="text-align: center; color: #B3B3B3; margin-top: 5px;">than average</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_pred3:
    st.markdown('<div class="spotify-metric">', unsafe_allow_html=True)
    
    # Estimated Popularity in text bar
    estimated_popularity = 50 + (difference / 2)
    estimated_popularity = max(0, min(100, estimated_popularity))
    pop_text = f"{estimated_popularity:.0f}"
    pop_color = "#1DB954" if estimated_popularity >= 70 else "#FFD700" if estimated_popularity >= 50 else "#E22134"
    
    st.markdown("##### ⭐ Estimated Popularity")
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {pop_color} 0%, #FFD700 100%);
        color: white;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
        font-size: 1.6em;
        font-weight: bold;
        font-family: 'Courier New', monospace;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        border: 2px solid white;
    ">
    {pop_text}/100
    </div>
    """, unsafe_allow_html=True)
    
    if estimated_popularity >= 70:
        rating = "Excellent"
    elif estimated_popularity >= 50:
        rating = "Good"
    else:
        rating = "Needs Work"
    
    st.markdown(f'<p style="text-align: center; color: #B3B3B3; margin-top: 5px;">{rating}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ====================
# 5. FEATURE IMPORTANCE VISUALIZATION
# ====================
st.markdown('<div class="spotify-card">', unsafe_allow_html=True)
st.markdown("## 📊 5. Feature Impact Analysis")

# Create feature importance chart with dark theme
feature_importance = pd.DataFrame({
    'Feature': X_features,
    'Impact Score': abs(model.coef_) / max(abs(model.coef_)) * 100
}).sort_values('Impact Score', ascending=True)

fig_bar = px.bar(feature_importance, 
                 x='Impact Score', 
                 y='Feature',
                 orientation='h',
                 title="Relative Impact of Each Feature on Streams",
                 color='Impact Score',
                 color_continuous_scale='viridis')
fig_bar.update_layout(
    plot_bgcolor='#212121',
    paper_bgcolor='#212121',
    font_color='white'
)
st.plotly_chart(fig_bar, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ====================
# 6. WHAT-IF SCENARIOS
# ====================
st.markdown('<div class="spotify-card">', unsafe_allow_html=True)
st.markdown("## 🔬 6. What-If Analysis")

what_if_col1, what_if_col2, what_if_col3 = st.columns(3)

with what_if_col1:
    st.markdown("### More Danceable")
    new_dance = danceability + 20
    pred_more_dance = predict_all_features(tempo, new_dance, acousticness, 
                                          loudness, energy, valence, duration)
    dance_impact = ((pred_more_dance - prediction) / prediction * 100)
    st.metric("+20 Danceability", f"{dance_impact:+.1f}% impact")

with what_if_col2:
    st.markdown("### Less Acoustic")
    new_acoustic = max(0, acousticness - 30)
    pred_less_acoustic = predict_all_features(tempo, danceability, new_acoustic,
                                             loudness, energy, valence, duration)
    acoustic_impact = ((pred_less_acoustic - prediction) / prediction * 100)
    st.metric("-30 Acousticness", f"{acoustic_impact:+.1f}% impact")

with what_if_col3:
    st.markdown("### Higher Energy")
    new_energy = min(100, energy + 20)
    pred_more_energy = predict_all_features(tempo, danceability, acousticness,
                                           loudness, new_energy, valence, duration)
    energy_impact = ((pred_more_energy - prediction) / prediction * 100)
    st.metric("+20 Energy", f"{energy_impact:+.1f}% impact")
st.markdown('</div>', unsafe_allow_html=True)

# ====================
# 7. OPTIMAL SONG RECOMMENDATION
# ====================
st.markdown('<div class="spotify-card">', unsafe_allow_html=True)
st.markdown("## 🏆 7. Optimal Hit Recipe")

# Calculate optimal values based on model
optimal_values = {
    'Tempo (BPM)': df['Tempo (BPM)'].iloc[df['Streams (billions)'].idxmax()],
    'Danceability (/100)': df['Danceability (/100)'].iloc[df['Streams (billions)'].idxmax()],
    'Acousticness (/100)': df['Acousticness (/100)'].iloc[df['Streams (billions)'].idxmax()],
    'Loudness (dB)': df['Loudness (dB)'].iloc[df['Streams (billions)'].idxmax()],
    'Energy (/100)': df['Energy (/100)'].iloc[df['Streams (billions)'].idxmax()],
    'Valence (/100)': df['Valence (/100)'].iloc[df['Streams (billions)'].idxmax()],
    'Duration (seconds)': df['Duration (seconds)'].iloc[df['Streams (billions)'].idxmax()]
}

st.markdown("**Based on top-performing songs in our dataset:**")
optimal_df = pd.DataFrame(list(optimal_values.items()), columns=['Feature', 'Optimal Value'])
st.table(optimal_df)

# Predict optimal song streams
optimal_prediction = predict_all_features(
    optimal_values['Tempo (BPM)'],
    optimal_values['Danceability (/100)'],
    optimal_values['Acousticness (/100)'],
    optimal_values['Loudness (dB)'],
    optimal_values['Energy (/100)'],
    optimal_values['Valence (/100)'],
    optimal_values['Duration (seconds)']
)

st.success(f"""
🎵 **Optimal Hit Song Profile:**
- Would generate approximately **{optimal_prediction:,.0f} streams**
- That's **{((optimal_prediction - avg_streams) / avg_streams * 100):+.1f}%** better than average
- Example: "Blinding Lights" by The Weeknd (3.5B streams)
""")
st.markdown('</div>', unsafe_allow_html=True)

# ====================
# 8. MODEL PERFORMANCE
# ====================
st.markdown('<div class="spotify-card">', unsafe_allow_html=True)
st.markdown("## 📋 8. Model Performance & Validation")

# Calculate R-squared
r_squared = model.score(X, y)

# Make predictions for all songs
predictions = model.predict(X)
actual = y

# Calculate performance metrics
mae = np.mean(np.abs(predictions - actual))
mse = np.mean((predictions - actual) ** 2)
rmse = np.sqrt(mse)

col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)

with col_perf1:
    st.markdown('<div class="spotify-metric">', unsafe_allow_html=True)
    st.markdown("##### R² Score")
    st.markdown(f"# {r_squared:.3f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col_perf2:
    st.markdown('<div class="spotify-metric">', unsafe_allow_html=True)
    st.markdown("##### Mean Absolute Error")
    st.markdown(f"# {mae:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col_perf3:
    st.markdown('<div class="spotify-metric">', unsafe_allow_html=True)
    st.markdown("##### Root Mean Squared Error")
    st.markdown(f"# {rmse:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col_perf4:
    accuracy = max(0, 100 * (1 - mae / avg_streams))
    st.markdown('<div class="spotify-metric">', unsafe_allow_html=True)
    st.markdown("##### Estimated Accuracy")
    st.markdown(f"# {accuracy:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ====================
# 9. LINEAR ALGEBRA INSIGHTS
# ====================
st.markdown('<div class="spotify-card">', unsafe_allow_html=True)
st.markdown("## 🧮 9. Linear Algebra Behind the Model")

st.markdown("""
**Mathematical Foundation:**

### 1. **Matrix Formulation:**
Our dataset is represented as matrices:
""")

st.latex(r'''
X = \begin{bmatrix}
\text{Tempo}_1 & \text{Danceability}_1 & \cdots & \text{Duration}_1 \\
\text{Tempo}_2 & \text{Danceability}_2 & \cdots & \text{Duration}_2 \\
\vdots & \vdots & \ddots & \vdots \\
\text{Tempo}_{20} & \text{Danceability}_{20} & \cdots & \text{Duration}_{20}
\end{bmatrix}
''')

st.latex(r'''
y = \begin{bmatrix}
\text{Streams}_1 \\
\text{Streams}_2 \\
\vdots \\
\text{Streams}_{20}
\end{bmatrix}
''')

st.markdown("""
### 2. **Normal Equation Solution:**
We solve for the coefficient vector β using:
""")

st.latex(r'''
\beta = (X^T X)^{-1} X^T y
''')

st.markdown("""
### 3. **Eigenvector Analysis:**
The covariance matrix reveals feature relationships:
""")

st.latex(r'''
C = X^T X
''')

st.markdown("""
The eigenvectors of C show the principal directions of variation in our data.
""")

# Calculate eigenvectors of the covariance matrix
X_scaled = StandardScaler().fit_transform(X)
cov_matrix = np.cov(X_scaled.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort eigenvalues and eigenvectors
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

st.markdown(f"""
### 4. **Principal Components:**
From our eigen decomposition:
- **First PC** explains **{eigenvalues[0]/sum(eigenvalues)*100:.1f}%** of variance
- **Second PC** explains **{eigenvalues[1]/sum(eigenvalues)*100:.1f}%** of variance
- **Top 3 PCs** explain **{sum(eigenvalues[:3])/sum(eigenvalues)*100:.1f}%** of total variance
""")

# Display top eigenvectors
st.markdown("**Top 2 Eigenvectors (Principal Components):**")

col_eig1, col_eig2 = st.columns(2)

with col_eig1:
    st.markdown("**PC1 - Main Pattern:**")
    pc1_df = pd.DataFrame({
        'Feature': X_features,
        'Loading': eigenvectors[:, 0]
    }).sort_values('Loading', key=abs, ascending=False)
    st.dataframe(pc1_df.head(5))

with col_eig2:
    st.markdown("**PC2 - Secondary Pattern:**")
    pc2_df = pd.DataFrame({
        'Feature': X_features,
        'Loading': eigenvectors[:, 1]
    }).sort_values('Loading', key=abs, ascending=False)
    st.dataframe(pc2_df.head(5))

st.markdown("""
### 5. **Interpretation:**
- **Positive loadings** indicate features that increase together
- **Negative loadings** indicate inverse relationships
- **Large absolute values** show strong influence on that principal direction
""")
st.markdown('</div>', unsafe_allow_html=True)

# ====================
# FOOTER
# ====================
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 30px; background: #121212; border-radius: 10px;">
    <h3 style="color: #1DB954;">🎵 Spotify Hit Predictor Pro</h3>
    <p style="color: #B3B3B3;"><strong>AMTH 222 Linear Algebra Project</strong> | Hussein Zindonda</p>
    <p style="color: #B3B3B3;"><strong>Dataset:</strong> Top 20 Spotify Songs with 11 Features</p>
    <p style="color: #B3B3B3;"><strong>Methods:</strong> Multiple Linear Regression • Correlation Analysis • PCA • Eigenvector Decomposition</p>
</div>
""", unsafe_allow_html=True)
