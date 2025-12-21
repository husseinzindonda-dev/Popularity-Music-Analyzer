import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Spotify Hit Predictor", layout="wide")
st.title("🎵 The Mathematics of a Hit Song")
st.subheader("AMTH 222 Project: Linear Algebra Analysis of Spotify Data")

# Load your data
@st.cache_data
def load_data():
    # Your data here (use the 30 songs from your project)
    data = pd.DataFrame({
        'song': ["Sunflower", "Lucid Dreams", ...],
        'tempo': [90, 84, ...],
        'streams': [883369738, 864832399, ...],
        'danceability': [76, 51, ...],
        'acousticness': [0.28, 0.177, ...]
    })
    return data

df = load_data()

# Sidebar controls
st.sidebar.header("🔧 Interactive Demo")
feature_x = st.sidebar.selectbox("X-Axis Feature", 
                                ['tempo', 'danceability', 'acousticness'])
feature_y = st.sidebar.selectbox("Y-Axis Feature", 
                                ['streams', 'danceability', 'acousticness'])

# Show covariance matrix
st.header("1. Feature Correlation Analysis")
corr_matrix = df[['tempo', 'danceability', 'acousticness']].corr()
fig = px.imshow(corr_matrix, text_auto=True, 
                labels=dict(x="Features", y="Features", color="Correlation"))
st.plotly_chart(fig, use_container_width=True)

# Interactive prediction
st.header("2. Stream Prediction Model")
col1, col2, col3 = st.columns(3)
with col1:
    tempo_input = st.slider("Tempo (BPM)", 60, 200, 120)
with col2:
    dance_input = st.slider("Danceability", 0, 100, 70)
with col3:
    acoustic_input = st.slider("Acousticness", 0.0, 1.0, 0.2)

# Your regression equation from the project
predicted_streams = 710387065.02 - 353581.02*tempo_input - 1229790.56*dance_input - 46423348.06*acoustic_input

st.metric("🎯 Predicted Streams", f"{predicted_streams:,.0f}")

# Eigenvector visualization
st.header("3. Eigenvector Analysis")
st.latex(r'''
\text{Top Eigenvector } \lambda=1.402: \quad 
\begin{bmatrix}
\text{Tempo} & \text{Streams} & \text{Danceability} & \text{Acousticness} \\
0.04 & -0.10 & 0.70 & -0.70
\end{bmatrix}
''')
st.caption("Negative correlation between Danceability and Acousticness")

# Actual vs Predicted Comparison
st.header("4. Model Performance")
# Add your predictions vs actual
st.dataframe(df[['song', 'streams']].head(10))