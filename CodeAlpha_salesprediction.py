# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


df = pd.read_csv('/Users/kudaykumar/Downloads/CodeAlpha/Datas sets/Advertising.csv')

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# Remove unnecessary index column
df.drop("Unnamed: 0", axis=1, inplace=True)


print("\nDataset Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

plt.figure(figsize=(8,5))
sns.histplot(df["Sales"], kde=True)

plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.show()


plt.figure(figsize=(8,5))

sns.scatterplot(
    x="TV",
    y="Sales",
    data=df
)

plt.title("TV Advertising vs Sales")

plt.show()

plt.figure(figsize=(8,5))

sns.scatterplot(
    x="Radio",
    y="Sales",
    data=df
)

plt.title("Radio Advertising vs Sales")

plt.show()

plt.figure(figsize=(8,5))

sns.scatterplot(
    x="Newspaper",
    y="Sales",
    data=df
)

plt.title("Newspaper Advertising vs Sales")

plt.show()


plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LinearRegression()

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

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.show()


importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nAdvertising Impact:")
print(importance)

plt.figure(figsize=(8,5))

sns.barplot(
    x="Coefficient",
    y="Feature",
    data=importance
)

plt.title("Impact of Advertising Channels on Sales")

plt.show()


sample_data = [[150, 25, 30]]

predicted_sales = model.predict(sample_data)

print("\nPredicted Sales:")
print(round(predicted_sales[0], 2))
print("\nBusiness Insights")
print("-" * 30)

print("1. Higher advertising budgets generally increase sales.")
print("2. TV advertising has the strongest influence on sales.")
print("3. Radio advertising also contributes significantly.")
print("4. Newspaper advertising has the least impact.")
print("5. Businesses should prioritize TV and Radio campaigns.")