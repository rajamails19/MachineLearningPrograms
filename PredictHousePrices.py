from sklearn.linear_model import LinearRegression
import numpy as np

# Example dataset: [Size of house in sqft, Price]
X = np.array([[1000], [1500], [2000], [2500], [3000]])
y = np.array([200000, 250000, 300000, 350000, 400000])

model = LinearRegression()
model.fit(X, y)

# Predict price for a house of size 1800 sqft
predicted_price = model.predict([[1800]])
print("Predicted Price: $", predicted_price[0])
