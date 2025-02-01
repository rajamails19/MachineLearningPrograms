from sklearn.linear_model import LogisticRegression
import numpy as np

# Example dataset: [Age, BMI], 0 = No Diabetes, 1 = Has Diabetes
X = np.array([[25, 22], [30, 28], [35, 30], [40, 32], [50, 35], [60, 40]])
y = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

# Predict diabetes for a person aged 45 with a BMI of 33
prediction = model.predict([[45, 33]])
print("Diabetes Prediction:", "Yes" if prediction[0] == 1 else "No")
