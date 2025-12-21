import numpy as np
from sklearn.linear_model import LinearRegression

# Feature matrix: [tempo, danceability, acousticness]
X = np.array([
    [90, 76, 0.28], [84, 51, 0.177], [155, 73, 0.002], [100, 70, 0.000],
    [160, 66, 0.124], [130, 84, 0.0038], [171, 52, 0.0015], [134, 87, 0.776],
    [75, 74, 0.26], [81, 55, 0.82], [150, 90, 0.001], [120, 69, 0.111],
    [150, 73, 0.001], [113, 78, 0.001], [123, 63, 0.22], [144, 56, 0.07],
    [118, 77, 0.44], [154, 75, 0.117], [117, 79, 0.00187], [95, 84, 0.15],
    [85, 80, 0.04], [145, 68, 0.33], [104, 89, 0.001], [169, 69, 0.34],
    [160, 72, 0.00398], [115, 35, 0.934], [89, 52, 0.184], [64, 92, 0.01],
    [100, 89, 0.0], [170, 59, 0.05]
])

# Target vector: streams
y = np.array([
    883369738, 864832399, 781153024, 734857487, 718865961, 672972704, 644287953,
    624457164, 619879245, 613872384, 606305588, 598521764, 586638599, 583443174,
    546036924, 543144261, 540754791, 534994242, 504210201, 481985952, 473417295,
    463551468, 454267392, 454100610, 443773199, 437911914, 437333177, 431568186,
    427614856, 426712325
])

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the coefficients
intercept = model.intercept_
tempo_coef, dance_coef, acoustic_coef = model.coef_

print("=== LEAST SQUARES REGRESSION RESULTS ===")
print(f"Intercept (β₀): {intercept:,.2f}")
print(f"Tempo coefficient (β₁): {tempo_coef:,.2f}")
print(f"Danceability coefficient (β₂): {dance_coef:,.2f}")
print(f"Acousticness coefficient (β₃): {acoustic_coef:,.2f}")

print(f"\nRegression Equation:")
print(f"Predicted Streams = {intercept:,.0f} + {tempo_coef:,.0f}×Tempo + {dance_coef:,.0f}×Danceability + {acoustic_coef:,.0f}×Acousticness")

print(f"\nR-squared: {model.score(X, y):.4f}")

# Predict for "The Box" by Roddy Ricch
the_box_features = np.array([[117, 79, 0.00187]])
the_box_prediction = model.predict(the_box_features)[0]

print(f"\n=== PREDICTION FOR 'THE BOX' ===")
print(f"Predicted streams: {the_box_prediction:,.0f}")
print(f"Actual streams: 504,210,201")
print(f"Difference: {504210201 - the_box_prediction:,.0f}")

# Interpretation
print(f"\n=== INTERPRETATION ===")
print(f"• Each 1 BPM increase → {tempo_coef:,.0f} more streams")
print(f"• Each 1-unit danceability increase → {dance_coef:,.0f} streams")
print(f"• Each 1-unit acousticness increase → {acoustic_coef:,.0f} streams")