# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


df = pd.read_csv('/Users/kudaykumar/Downloads/CodeAlpha/Datas sets/car data.csv')

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# Calculate Car Age
current_year = 2025
df["Car_Age"] = current_year - df["Year"]

# Remove unnecessary columns
df.drop(["Car_Name", "Year"], axis=1, inplace=True)


# 1. Selling Price Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Selling_Price"], kde=True)
plt.title("Distribution of Selling Price")
plt.show()

# 2. Correlation Heatmap

encoded_df = df.copy()

encoded_df["Fuel_Type"] = encoded_df["Fuel_Type"].astype("category").cat.codes
encoded_df["Selling_type"] = encoded_df["Selling_type"].astype("category").cat.codes
encoded_df["Transmission"] = encoded_df["Transmission"].astype("category").cat.codes

plt.figure(figsize=(10,6))
sns.heatmap(
    encoded_df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()


df = pd.get_dummies(
    df,
    columns=[
        "Fuel_Type",
        "Selling_type",
        "Transmission"
    ],
    drop_first=True
)


X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)



y_pred = model.predict(X_test)



mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-" * 30)

print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 4))


plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")

plt.show()


importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance.sort_values().plot(
    kind="barh",
    figsize=(10,6)
)

plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.show()
sample_car = X.iloc[[0]]

predicted_price = model.predict(sample_car)

print("\nSample Predicted Price:")
print(round(predicted_price[0], 2), "Lakhs")