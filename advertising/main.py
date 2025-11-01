import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("advertising.csv")  # path to your dataset

# Features and target
X = df[["TV", "Radio", "Newspaper"]]  # predictors
y = df["Sales"]                       # target

# Fit model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R²:", r2)



# THIS IS THE IMPLEMENTATION FROM SCRATCH
####################################################

df = pd.read_csv("advertising.csv")

# Extract X and y
x = df["TV"].values
y = df["Sales"].values

# Compute means
x_mean = np.mean(x)
y_mean = np.mean(y)

# Compute m (slope)
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean)**2)
m = numerator / denominator

# Compute b (intercept)
b = y_mean - m * x_mean

# print("Slope (m):", m)
# print("Intercept (b):", b)

# Predict function
def predict(x):
    return m*x + b

####################################################

# MULTI REGRSSION FOR AD DATASET

# Load the advertising dataset
df = pd.read_csv("advertising.csv")

# Extract input features and target
X = df[['TV', 'Radio', 'Newspaper']].values
y = df['Sales'].values.reshape(-1, 1)

# Add bias term (column of 1s for b0/intercept)
X_b = np.hstack([np.ones((X.shape[0], 1)), X])

# Compute coefficients using OLS formula: (X^T X)^(-1) X^T y
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

b0, b1, b2, b3 = theta.flatten()

print(f"Intercept (b0): {b0}")
print(f"TV coefficient (b1): {b1}")
print(f"Radio coefficient (b2): {b2}")
print(f"Newspaper coefficient (b3): {b3}")


# Example: input marketing spend
TV = 100
Radio = 25
Newspaper = 20

# Prediction using our coefficients
sales_pred = b0 + b1*TV + b2*Radio + b3*Newspaper
print(sales_pred)

# running R2 for linear regression

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Reload data
df = pd.read_csv("advertising.csv")

# Model 1: Sales ~ TV only
X = df[["TV"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X, y)

# Predictions & R^2
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print(r2)

####################################################
# code for mae, mse, r2

# Load dataset
df = pd.read_csv("advertising.csv")  # path to your dataset

# Features and target
X = df[["TV", "Radio", "Newspaper"]]  # predictors
y = df["Sales"]                       # targetc

# Fit model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R²:", r2)

