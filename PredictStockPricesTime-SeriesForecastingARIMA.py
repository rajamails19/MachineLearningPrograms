import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# Generate sample time-series data (simulated stock prices)
dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
stock_prices = np.cumsum(np.random.randn(100) * 2) + 100
df = pd.DataFrame({'Date': dates, 'Price': stock_prices})

# Train ARIMA model
model = ARIMA(df['Price'], order=(5,1,0))
model_fit = model.fit()

# Predict next 10 days
forecast = model_fit.forecast(steps=10)
future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=11, freq="D")[1:]

# Plot results
plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['Price'], label="Historical Prices")
plt.plot(future_dates, forecast, label="Predicted Prices", linestyle="dashed", color="red")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
