from sklearn.linear_model import LinearRegression
import numpy as np

# Example dataset: [Study Hours, Sleep Hours], Exam Score
X = np.array([[2, 8], [3, 7], [5, 6], [6, 6], [7, 5], [8, 5]])
y = np.array([50, 55, 70, 75, 80, 85])

model = LinearRegression()
model.fit(X, y)

# Predict score for a student who studied 6 hours and slept 7 hours
predicted_score = model.predict([[6, 7]])
print("Predicted Exam Score:", predicted_score[0])
