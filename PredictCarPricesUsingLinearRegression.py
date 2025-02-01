from sklearn.linear_model import LinearRegression
import numpy as np

# Example dataset: [Mileage in km, Price of car]
X = np.array([[5000], [15000], [25000], [40000], [60000], [80000]])
y = np.array([30000, 25000, 20000, 15000, 10000, 5000])

model = LinearRegression()
model.fit(X, y)

# Predict price for a car with 35000 km mileage
predicted_price = model.predict([[35000]])
print("Predicted Car Price: $", predicted_price[0])
