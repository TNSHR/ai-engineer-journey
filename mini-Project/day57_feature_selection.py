import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Dataset
np.random.seed(42)

size = 200

marks = np.random.randint(30, 100, size)
attendance = np.random.randint(50, 100, size)
study_hours = np.random.randint(1, 8, size)
previous_score = np.random.randint(30, 100, size)

# Noise feature
random_noise = np.random.randint(0, 100, size)

at_risk = (marks < 50).astype(int)

df = pd.DataFrame({
    "marks": marks,
    "attendance": attendance,
    "study_hours": study_hours,
    "previous_score": previous_score,
    "random_noise": random_noise,
    "at_risk": at_risk
})

X = df.drop("at_risk", axis=1)
y = df["at_risk"]

# Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

# Model
model = RandomForestClassifier(random_state=42)

model.fit(X_train,y_train)

# Feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print(importance.sort_values(by="Importance", ascending=False))