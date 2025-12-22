# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Spotify Hit Analyzer", layout="wide")

# Title
st.title("🎵 The Mathematics of a Hit Song")
st.subheader("AMTH 222 Project: Spotify Data Analysis with Linear Algebra")

# Sidebar with team info
with st.sidebar:
    st.header("👥 Team Info")
    st.markdown("**Oscar De La Cerda**")
    st.markdown("**Hussein Zindonda**")
    st.markdown("**Sofia Colorado**")
    st.markdown("---")
    st.markdown("**Course:** AMTH 222")
    st.markdown("**Instructor:** Tamunonye Cheetham-West")
    st.markdown("**Date:** December 2025")

# ====================
# HARDCODED DATA (Fix for Streamlit)
# ====================
data = {
    'tempo': [90, 84, 155, 100, 160, 130, 171, 134, 75, 81, 150, 120, 150, 113, 123, 144, 118, 154, 117, 95, 85, 145, 104, 169, 160, 115, 89, 64, 100, 170],
    'streams': [883369738, 864832399, 781153024, 734857487, 718865961, 672972704, 644287953, 624457164, 619879245, 613872384, 606305588, 598521764, 586638599, 583443174, 546036924, 543144261, 540754791, 534994242, 504210201, 481985952, 473417295, 463551468, 454267392, 454100610, 443773199, 437911914, 437333177, 431568186, 427614856, 426712325],
    'danceability': [76, 51, 73, 70, 66, 84, 52, 87, 74, 55, 90, 69, 73, 78, 63, 56, 77, 75, 79, 84, 80, 68, 89, 69, 72, 35, 52, 92, 89, 59],
    'acousticness': [0.28, 0.177, 0.002, 0.000, 0.124, 0.0038, 0.0015, 0.776, 0.26, 0.82, 0.001, 0.111, 0.001, 0.001, 0.22, 0.07, 0.44, 0.117, 0.00187, 0.15, 0.04, 0.33, 0.001, 0.34, 0.00398, 0.934, 0.184, 0.01, 0.0, 0.05]
}

df = pd.DataFrame(data)

# ====================
# 1. CORRELATION ANALYSIS
# ====================
st.header("📈 1. Feature Correlation Analysis")

# Calculate correlation matrix (THIS WILL NOW WORK)
corr_matrix = df[['tempo', 'danceability', 'acousticness']].corr()

# Display correlation matrix
st.subheader("Correlation Matrix")
fig = px.imshow(corr_matrix, 
                text_auto=True,
                color_continuous_scale='RdBu_r',
                aspect="auto",
                title="Feature Correlations")
st.plotly_chart(fig, use_container_width=True)

# Key insight
st.info("""
🔍 **Key Finding:** There's a **negative correlation between danceability and acousticness** (-0.398).
This means popular songs tend to be either danceable OR acoustic, rarely both.
""")

# ====================
# 2. EIGENVECTOR ANALYSIS
# ====================
st.header("🧮 2. Eigenvector Analysis (PCA)")

# Calculate eigenvectors
C = df[['tempo', 'danceability', 'acousticness']].values
means = C.mean(axis=0)
stds = C.std(axis=0)
C_z = (C - means) / stds
C_col_norm = C_z / np.linalg.norm(C_z, axis=0, keepdims=True)
feature_similarity = C_col_norm.T @ C_col_norm
eigenvalues, eigenvectors = np.linalg.eigh(feature_similarity)

# Display results
col1, col2 = st.columns(2)

with col1:
    st.subheader("Eigenvalues")
    for i, val in enumerate(sorted(eigenvalues, reverse=True)):
        st.metric(f"λ_{i+1}", f"{val:.3f}")

with col2:
    st.subheader("Dominant Eigenvector")
    st.markdown(f"""
    **λ = {sorted(eigenvalues, reverse=True)[0]:.3f}** (Largest eigenvalue)
    
    Direction: {eigenvectors[:,-1].round(3)}
    
    **Interpretation:** This eigenvector shows the strongest pattern in the data.
    The negative relationship between danceability and acousticness is the most significant feature relationship.
    """)

# ====================
# 3. STREAM PREDICTION
# ====================
st.header("🎯 3. Stream Prediction Model")

st.markdown("""
**Regression Equation from our analysis:**
""")
st.latex(r'''
\text{Predicted Streams} = 710,\!387,\!065.02 - 353,\!581.02 \times \text{Tempo} - 1,\!229,\!790.56 \times \text{Danceability} - 46,\!423,\!348.06 \times \text{Acousticness}
''')

# Interactive prediction
st.subheader("Interactive Predictor")

col1, col2, col3 = st.columns(3)

with col1:
    tempo_input = st.slider("🎵 Tempo (BPM)", 60, 200, 120, 5)
    st.metric("Selected", f"{tempo_input} BPM")

with col2:
    dance_input = st.slider("💃 Danceability", 0, 100, 70, 1)
    st.metric("Selected", f"{dance_input}/100")

with col3:
    acoustic_input = st.slider("🎸 Acousticness", 0.0, 1.0, 0.2, 0.05)
    st.metric("Selected", f"{acoustic_input:.2f}")

# Prediction function
def predict_streams(tempo, dance, acoustic):
    return 710387065.02 - 353581.02*tempo - 1229790.56*dance - 46423348.06*acoustic

# Calculate and display prediction
predicted = predict_streams(tempo_input, dance_input, acoustic_input)

st.markdown("---")
st.subheader("📊 Prediction Result")
col_pred1, col_pred2 = st.columns(2)

with col_pred1:
    st.metric("Predicted Streams", f"{predicted:,.0f}")

with col_pred2:
    # Compare with average
    avg_streams = df['streams'].mean()
    difference = ((predicted - avg_streams) / avg_streams * 100)
    st.metric("Vs. Average", f"{difference:+.1f}%")

# ====================
# 4. MODEL PERFORMANCE
# ====================
st.header("📋 4. Model Performance")

# Test case: Drake's "One Dance"
st.subheader("Test Case: Drake - 'One Dance'")
test_data = pd.DataFrame({
    'Metric': ['Actual Streams', 'Predicted Streams', 'Error'],
    'Value': ['454,267,392', '564,116,855', '24.2%']
})
st.table(test_data)

st.success("""
✅ **Model successfully identifies acousticness as the strongest negative predictor of streams.**  
✅ **Eigenvector analysis reveals key feature relationships in popular music.**  
✅ **Interactive tool demonstrates practical application of linear algebra.**
""")

# ====================
# FOOTER
# ====================
st.markdown("---")
st.caption("AMTH 222 Linear Algebra Project | Analysis of Spotify's Top 30 Streamed Songs")
