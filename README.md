## 🎵 The Mathematics of a Hit Song: Spotify Data Analysis
## 📊 Project Overview
An academic project applying linear algebra concepts to analyze Spotify's top-streamed songs. We explore the mathematical relationships between song features (tempo, danceability, acousticness) and popularity (streams) using covariance matrices, eigenvectors, and least squares regression.
Course: AMTH 222 - Linear Algebra
Team: Oscar De La Cerda, Hussein Zindonda, Sofia Colorado
Instructor: Tamunonye Cheetham-West
Date: December 2025
## 🎯 Research Questions
1. Which song features most strongly correlate with popularity among top-charting songs?
2. Can we predict a song's stream count based on its musical features?
## 📁 Project Structure
```
spotify-linear-analysis/
│
├── 📄 README.md # This file
├── 📄 AMTH 222 project.docx # Project report
├── 📄 222_project.py # Python analysis code
└── 📄 spotify_top30.csv # Dataset (30 songs)
```
## 🔍 Key Findings
### 📈 Feature Correlation Analysis
- Strongest Pattern (λ = 1.402): Negative correlation between danceability and acousticness
- Secondary Pattern (λ = 1.095): Tempo varies independently across top songs
- Insight: Popular songs tend to be either danceable OR acoustic, rarely both
### 🤖 Stream Prediction Model
Regression Equation:
```
Predicted Streams = 710,387,065.02
- 353,581.02 × Tempo
- 1,229,790.56 × Danceability
- 46,423,348.06 × Acousticness
```
Model Performance:
- Tested on Drake's "One Dance": Predicted vs Actual error ≈ 24%
- Key predictor: Acousticness (largest coefficient magnitude)
## 📊 Data Source
Dataset: Spotify Top 10000 Streamed Songs
Sample Size: Top 20 songs from the dataset
Features Used:
- tempo (beats per minute)
- streams (total plays on Spotify)
- danceability (0-100 scale)
- acousticness (0-1 scale)
## 🧮 Mathematical Methods
### 1. Correlation Analysis
- Tool: Principal Component Analysis (PCA)
- Process:
1. Z-score normalization of features
2. Compute covariance matrix: C = AᵀA
3. Eigen decomposition: Cv = λv
- Output: Eigenvectors showing which feature combinations explain the most variance
### 2. Stream Prediction
- Tool: Multiple Linear Regression
- Method: Least Squares via Normal Equation
- Equation: (XᵀX)β = Xᵀy
- Implementation: Both direct matrix inversion and scikit-learn's QR decomposition
## 📈 Results Summary
| Analysis Type | Key Insight | Mathematical Tool |
|--------------|-------------|-------------------|
| Feature Correlation | Danceability & acousticness are inversely related | Eigenvectors of covariance matrix |
| Popularity Prediction | Acousticness has strongest negative impact on streams | Least squares regression |
| Model Validation | 24% error on test song prediction | Residual analysis |

Eigenvector Results:
- λ = 1.402: Shows strong negative relationship between danceability and acousticness
- λ = 1.095: Indicates tempo varies independently of other features
- λ = 1.009 & 0.492: Represent less significant patterns in the data
Regression Coefficients:
- Intercept: 710,387,065.02
- Tempo: -353,581.02 (negative effect)
- Danceability: -1,229,790.56 (negative effect)
- Acousticness: -46,423,348.06 (strongest negative effect)
## 👥 Team Contributions
| Team Member | Primary Contribution |
|-------------|---------------------|
| Oscar | Regression analysis, mathematical formulation |
| Hussein | Data processing, covariance matrix implementation |
| Sofia | Report writing, documentation, project coordination |
All team members contributed to code development, analysis, and interpretation of results.
## 📚 References
1. Strang, G. (2016). Introduction to Linear Algebra (5th ed.)
2. Spotify API Documentation: Audio Features
3. Kaggle Dataset: Spotify Top 10000 Streamed Songs
4. Python Libraries: NumPy, scikit-learn
## 📄 Academic Context
This project demonstrates practical applications of linear algebra concepts learned in AMTH 222:
- Covariance matrices for understanding multivariate relationships
- Eigen decomposition for dimensionality reduction (PCA)
- Least squares for predictive modeling
- Matrix operations in Python for computational efficiency
## 🙏 Acknowledgments
- Tamunonye Cheetham-West for project guidance and instruction
- Spotify for making streaming data publicly available
- Kaggle community for dataset curation
- Python open-source community for scientific computing tools
## ⚠️ Limitations & Future Work
Current Limitations:
- Small sample size (20 songs)
- Limited feature set
- Linear model may not capture complex relationships
Potential Extensions:
- Analyze full dataset of 10,000 songs
- Include additional features 
- Try non-linear models or regularization techniques
---
Project Status: Complete
Course: AMTH 222 - Linear Algebra
Last Updated: December 2, 2025
---
For questions about this project, please contact the team members directly. This project was completed for academic purposes as part of AMTH 222 coursework.
