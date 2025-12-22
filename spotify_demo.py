# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Spotify Hit Predictor Pro", layout="wide")

# Title
st.title("🎵 Spotify Hit Predictor Pro")
st.subheader("Multi-Feature Linear Algebra Analysis - AMTH 222 Project")

# Sidebar with team info
with st.sidebar:
    st.header("Class Info")
    st.markdown("**Hussein Zindonda**")
    st.markdown("---")
    st.markdown("**Course:** AMTH 222")
    st.markdown("**Instructor:** Tamunonye Cheetham-West")
    st.markdown("**Date:** December 2025")

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
st.header("📈 1. Comprehensive Correlation Analysis")

# Calculate full correlation matrix
numerical_features = ['Tempo (BPM)', 'Popularity (/100)', 'Streams (billions)', 
                      'Danceability (/100)', 'Acousticness (/100)', 'Loudness (dB)',
                      'Duration (seconds)', 'Energy (/100)', 'Valence (/100)',
                      'Streams per Second', 'Energy per Loudness']

corr_matrix = df[numerical_features].corr()

# Display correlation matrix
fig = px.imshow(corr_matrix, 
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                aspect="auto",
                title="Comprehensive Feature Correlation Matrix")
st.plotly_chart(fig, use_container_width=True)

# Key insights
col1, col2, col3 = st.columns(3)
with col1:
    st.info("""
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
    st.success("""
    **New Insights:**
    - Streams/Second ↔ Popularity: +0.74
    - Duration ↔ Streams/Second: -0.53
    - Energy ↔ Acousticness: -0.74
    """)

# ====================
# 2. MULTI-FEATURE LINEAR REGRESSION
# ====================
st.header("🎯 2. Multi-Feature Stream Prediction Model")

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

# Display coefficients
coeff_df = pd.DataFrame({
    'Feature': X_features,
    'Coefficient': model.coef_,
    'Impact': ['Positive' if c > 0 else 'Negative' for c in model.coef_],
    'Magnitude': [f"{abs(c):,.0f}" for c in model.coef_]
})
st.dataframe(coeff_df.style.format({'Coefficient': '{:,.0f}'}))

# ====================
# 3. INTERACTIVE PREDICTOR WITH ALL FEATURES
# ====================
st.header("🎛️ 3. Interactive Song Builder")

st.markdown("**Adjust all features to build your perfect hit song:**")

# Create two rows of sliders
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("🎵 Rhythm")
    tempo = st.slider("Tempo (BPM)", 60, 200, 120, 5)
    duration = st.slider("Duration (sec)", 60, 360, 200, 10)
    
with col2:
    st.subheader("💃 Feel")
    danceability = st.slider("Danceability", 0, 100, 70, 1)
    valence = st.slider("Valence (Positivity)", 0, 100, 60, 1)
    
with col3:
    st.subheader("🔊 Sound")
    acousticness = st.slider("Acousticness", 0, 100, 20, 1)
    loudness = st.slider("Loudness (dB)", -10, 0, -5, 1)
    
with col4:
    st.subheader("⚡ Energy")
    energy = st.slider("Energy", 0, 100, 80, 1)

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

# ====================
# 4. PREDICTION ENGINE
# ====================
st.header("📊 4. Stream Prediction")

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

# Display results
col_pred1, col_pred2, col_pred3 = st.columns(3)

with col_pred1:
    st.metric("🎯 Predicted Streams", f"{prediction:,.0f}")

with col_pred2:
    st.metric("📈 Vs. Average", f"{difference:+.1f}%")

with col_pred3:
    # Estimate popularity (based on correlation)
    estimated_popularity = 50 + (difference / 2)
    estimated_popularity = max(0, min(100, estimated_popularity))
    st.metric("⭐ Estimated Popularity", f"{estimated_popularity:.0f}/100")

# ====================
# 5. FEATURE IMPORTANCE VISUALIZATION
# ====================
st.header("📊 5. Feature Impact Analysis")

# Create feature importance chart
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
st.plotly_chart(fig_bar, use_container_width=True)

# ====================
# 6. WHAT-IF SCENARIOS
# ====================
st.header("🔬 6. What-If Analysis")

what_if_col1, what_if_col2, what_if_col3 = st.columns(3)

with what_if_col1:
    st.subheader("More Danceable")
    new_dance = danceability + 20
    pred_more_dance = predict_all_features(tempo, new_dance, acousticness, 
                                          loudness, energy, valence, duration)
    dance_impact = ((pred_more_dance - prediction) / prediction * 100)
    st.metric("+20 Danceability", f"{dance_impact:+.1f}% impact")

with what_if_col2:
    st.subheader("Less Acoustic")
    new_acoustic = max(0, acousticness - 30)
    pred_less_acoustic = predict_all_features(tempo, danceability, new_acoustic,
                                             loudness, energy, valence, duration)
    acoustic_impact = ((pred_less_acoustic - prediction) / prediction * 100)
    st.metric("-30 Acousticness", f"{acoustic_impact:+.1f}% impact")

with what_if_col3:
    st.subheader("Higher Energy")
    new_energy = min(100, energy + 20)
    pred_more_energy = predict_all_features(tempo, danceability, acousticness,
                                           loudness, new_energy, valence, duration)
    energy_impact = ((pred_more_energy - prediction) / prediction * 100)
    st.metric("+20 Energy", f"{energy_impact:+.1f}% impact")

# ====================
# 7. OPTIMAL SONG RECOMMENDATION
# ====================
st.header("🏆 7. Optimal Hit Recipe")

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

# ====================
# 8. MODEL PERFORMANCE
# ====================
st.header("📋 8. Model Performance & Validation")

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
    st.metric("R² Score", f"{r_squared:.3f}")

with col_perf2:
    st.metric("Mean Absolute Error", f"{mae:,.0f}")

with col_perf3:
    st.metric("Root Mean Squared Error", f"{rmse:,.0f}")

with col_perf4:
    accuracy = max(0, 100 * (1 - mae / avg_streams))
    st.metric("Estimated Accuracy", f"{accuracy:.1f}%")

# ====================
# 9. LINEAR ALGEBRA INSIGHTS
# ====================
st.header("🧮 9. Linear Algebra Behind the Model")

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

# ====================
# FOOTER
# ====================
st.markdown("---")
st.markdown("""
**AMTH 222 Linear Algebra Project**  
**Dataset:** Top 20 Spotify Songs with 11 Features  
**Methods Used:** Multiple Linear Regression, Correlation Analysis, PCA, Eigenvector Decomposition  
**Tools:** NumPy, pandas, scikit-learn, Streamlit
""")


