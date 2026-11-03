
# HOUSE PRICE PREDICTION PROJECT

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load dataset
data = pd.read_csv("C:/Users/hp/Downloads/housing dataset.csv")

print("First five rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nDataset information:")
data.info()

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate rows:")
print(data.duplicated().sum())

# 3. Remove duplicates and rows without price
data = data.drop_duplicates()
data = data.dropna(subset=["Price"])

# 4. EDA - Price distribution
sns.histplot(data["Price"], kde=True)
plt.title("House Price Distribution")
plt.show()

# 5. EDA - Area and price
sns.scatterplot(data=data, x="area", y="Price")
plt.title("Area vs House Price")
plt.show()

# 6. EDA - Correlation
sns.heatmap(
    data.select_dtypes(include="number").corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# 7. Select input and output
X = data[["area", "Year", "Rooms", "Floor", "Location"]].copy()
y = data["Price"].copy()

# 8. Fill missing numeric values
num_columns = ["area", "Year", "Rooms", "Floor"]

num_imputer = SimpleImputer(strategy="median")
X[num_columns] = num_imputer.fit_transform(X[num_columns])

# Fill missing locations
X["Location"] = X["Location"].fillna(
    X["Location"].mode()[0]
)

# Convert Location into numeric columns
X = pd.get_dummies(X, columns=["Location"], dtype=int)

# 9. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 10. Scale the data for polynomial regression and SVR
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 11. Model 1 - Simple Linear Regression
simple_model = LinearRegression()

simple_model.fit(
    X_train[["area"]], y_train
)

simple_pred = simple_model.predict(
    X_test[["area"]]
)

# 12. Model 2 - Multiple Linear Regression
multiple_model = LinearRegression()

multiple_model.fit(X_train, y_train)

multiple_pred = multiple_model.predict(X_test)

# 13. Model 3 - Polynomial Regression
poly = PolynomialFeatures(
    degree=2, include_bias=False
)

X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_model = LinearRegression()

poly_model.fit(X_train_poly, y_train)

poly_pred = poly_model.predict(X_test_poly)

# 14. Model 4 - Decision Tree Regression
tree_model = DecisionTreeRegressor(
    max_depth=5, random_state=42
)

tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)

# 15. Model 5 - Support Vector Regression
svr_model = SVR(kernel="rbf")

svr_model.fit(X_train_scaled, y_train)

svr_pred = svr_model.predict(X_test_scaled)

# 16. Evaluate all models
predictions = [
    ("Simple Linear Regression", simple_pred),
    ("Multiple Linear Regression", multiple_pred),
    ("Polynomial Regression", poly_pred),
    ("Decision Tree Regression", tree_pred),
    ("Support Vector Regression", svr_pred)
]

results = []

print("\nMODEL EVALUATION")

for name, pred in predictions:
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)

    print("\nModel:", name)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

# 17. Compare all models
results_df = pd.DataFrame(results)

print("\nFINAL COMPARISON")
print(
    results_df.sort_values(
        by="R2 Score", ascending=False
    ).to_string(index=False)
)

# 18. Plot model comparison
plt.figure(figsize=(10, 5))

plt.bar(
    results_df["Model"],
    results_df["R2 Score"]
)

plt.title("Comparison of Five Models")
plt.ylabel("R2 Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 19. Find the best model by R2 score
best_model = results_df.loc[
    results_df["R2 Score"].idxmax()
]

print("\nBest model based on R2 score:")
print(best_model["Model"])

# 20. Display one sample prediction
sample_index = 0

print("\nActual house price:", y_test.iloc[sample_index])
print("Predicted prices:")

for name, pred in predictions:
    print(name, ":", pred[sample_index])


