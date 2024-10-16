### Developed by : RAGUNATH R
### Register no : 212222240081
### Date: 

# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING

### AIM:
To implement Moving Average Model and Exponential smoothing Using Astrobiological dataset.
### ALGORITHM:
1. Import necessary libraries
2. Read the temperature time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the dataset
df = pd.read_csv('/content/dataset.csv')

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Convert 'Daily minimum temperatures' column to numeric, if not already
df['Daily minimum temperatures'] = pd.to_numeric(df['Daily minimum temperatures'], errors='coerce')

# Drop any NaN values
df.dropna(inplace=True)

# Display shape and first few rows of the dataset
print("Dataset shape:", df.shape)
print("First rows of dataset:\n", df.head(20))

# Plot the original data ('Daily minimum temperatures')
plt.figure(figsize=(10, 6))
plt.plot(df['Daily minimum temperatures'], label='Original Data', marker='o')
plt.title('Original Time Series Data (Daily Minimum Temperatures)')
plt.ylabel('Temperature (°C)')
plt.xlabel('Date')
plt.legend()
plt.show()

# Moving Average with window size 5 and 10
rolling_mean_5 = df['Daily minimum temperatures'].rolling(window=5).mean()
rolling_mean_10 = df['Daily minimum temperatures'].rolling(window=10).mean()

# Plot original data and rolling means (5 and 10)
plt.figure(figsize=(10, 6))
plt.plot(df['Daily minimum temperatures'], label='Original Data', marker='o')
plt.plot(rolling_mean_5, label='Rolling Mean (Window=5)', marker='x')
plt.plot(rolling_mean_10, label='Rolling Mean (Window=10)', marker='^')
plt.title('Original Data vs Rolling Means')
plt.ylabel('Temperature (°C)')
plt.xlabel('Date')
plt.legend()
plt.show()

# Perform Exponential Smoothing
exp_smoothing = SimpleExpSmoothing(df['Daily minimum temperatures']).fit(smoothing_level=0.2, optimized=False)
exp_smoothed = exp_smoothing.fittedvalues

# Plot Original Data and Exponential Smoothing
plt.figure(figsize=(10, 6))
plt.plot(df['Daily minimum temperatures'], label='Original Data', marker='o')
plt.plot(exp_smoothed, label='Exponential Smoothing', marker='s')
plt.title('Original Data vs Exponential Smoothing')
plt.ylabel('Temperature (°C)')
plt.xlabel('Date')
plt.legend()
plt.show()

# Plot ACF and PACF
plt.figure(figsize=(10, 6))
plt.subplot(121)
plot_acf(df['Daily minimum temperatures'], lags=10, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')
plt.subplot(122)
plot_pacf(df['Daily minimum temperatures'], lags=10, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# Generate Predictions using Exponential Smoothing (Predict next 3 values)
prediction_steps = 3
forecast = exp_smoothing.forecast(steps=prediction_steps)

# Plot original data and predictions
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Daily minimum temperatures'], label='Original Data', marker='o')
plt.plot(pd.date_range(start=df.index[-1], periods=prediction_steps + 1, freq='D')[1:], forecast, label='Predictions', marker='x')
plt.title('Original Data vs Predictions (Exponential Smoothing)')
plt.ylabel('Temperature (°C)')
plt.xlabel('Date')
plt.legend()
plt.show()
```

### OUTPUT:

![WhatsApp Image 2024-10-16 at 14 12 56_103d3ec1](https://github.com/user-attachments/assets/21e1d3ae-3a5e-4efa-9f23-72a2620bc4ee)

![WhatsApp Image 2024-10-16 at 14 13 40_7a28f125](https://github.com/user-attachments/assets/7ef0f7e3-dfe7-47e4-87c6-c68f6214a1e2)

![WhatsApp Image 2024-10-16 at 14 13 59_7ac759f7](https://github.com/user-attachments/assets/bc793ba8-59fb-47ee-846e-43b3e2de6d8f)

![WhatsApp Image 2024-10-16 at 14 14 49_47d8f2c1](https://github.com/user-attachments/assets/cbf647f4-ca12-4205-a178-734163f375ba)

![WhatsApp Image 2024-10-16 at 14 15 19_cd4e3b09](https://github.com/user-attachments/assets/cb0fcef2-b944-4137-b9d8-2577f38eb83a)

![WhatsApp Image 2024-10-16 at 14 15 47_d90ccbb3](https://github.com/user-attachments/assets/bb16b65f-5d61-4033-a46f-871da24a9376)



### RESULT:
Thus the python code successfully implemented for the Moving Average Model and Exponential smoothing  for daily minimum temperature dataset.
