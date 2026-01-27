import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
data = {
    "experience_years": [1, 2, 3, 4, 5, 6, 7, 8],
    "hours_per_week": [35,40,45,40,50,45,55,60],
    "salary_lpa":[3,4,5,6,8,9,11,13]
}

df = pd.DataFrame(data)
print("Dataset:\n", df)

# Features and target variable
X = df[["experience_years", "hours_per_week"]]
y = df["salary_lpa"]

# Train-test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state= 42
)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
print("Predicted Salaries:", predictions)
print("Actual Salaries:", y_test.values)

# Model Evaluation
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("MAE:", mae)
print("MSE:", mse)