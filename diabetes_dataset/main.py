# import pandas as pd
# from sklearn.datasets import load_diabetes
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# import numpy as np
# import matplotlib.pyplot as plt

# # Load diabetes dataset
# data = load_diabetes()

# print(data.data)

# X = pd.DataFrame(data.data, columns=data.feature_names)
# y = pd.Series(data.target, name="disease_progression")

# # Fit linear regression model
# model = LinearRegression().fit(X, y)
# y_pred = model.predict(X)

# print()

# # Metrics
# r2 = r2_score(y, y_pred)
# mae = mean_absolute_error(y, y_pred)
# mse = mean_squared_error(y, y_pred)
# rmse = np.sqrt(mse)

# print("R²:", r2)
# print("MAE:", mae)
# print("MSE:", mse)
# print("RMSE:", rmse)

# # Plot actual vs predicted
# plt.figure()
# plt.scatter(y, y_pred)
# plt.plot([y.min(), y.max()], [y.min(), y.max()])  # perfect prediction line
# plt.xlabel("Actual Disease Progression")
# plt.ylabel("Predicted Disease Progression")
# plt.title("Actual vs Predicted (Diabetes - Linear Regression)")
# plt.show()

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

# Load dataset
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train/Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("R² (test):", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)