import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Create dataset
np.random.seed(42)

size = 200

marks = np.random.randint(30, 100, size)
attendance = np.random.randint(50, 100, size)
study_hours = np.random.randint(1, 8, size)
previous_score = np.random.randint(30, 100, size)

at_risk = (marks < 50).astype(int)

# Add noise
noise_idx = np.random.choice(size, 30)
at_risk[noise_idx] = 1 - at_risk[noise_idx]

df = pd.DataFrame({
    "marks": marks,
    "attendance": attendance,
    "study_hours": study_hours,
    "previous_score": previous_score,
    "at_risk": at_risk
})

# Feature Engineering
df["effort_ratio"] = df["study_hours"] / df["attendance"]
df["performance_index"] = (df["marks"] + df["previous_score"]) / 2

X = df.drop("at_risk", axis=1)
y = df["at_risk"]

# Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

# Model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train,y_train)

y_pred = pipeline.predict(X_test)

# Combine results
result = X_test.copy()
result["Actual"] = y_test.values
result["Predicted"] = y_pred

# Find mistakes
errors = result[result["Actual"] != result["Predicted"]]

print("\nTotal Errors:", len(errors))
print("\nSample Errors:\n", errors.head())