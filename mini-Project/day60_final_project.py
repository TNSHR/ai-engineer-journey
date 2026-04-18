import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report

# 1️⃣ Create dataset
np.random.seed(42)

size = 500

marks = np.random.randint(30, 100, size)
attendance = np.random.randint(50, 100, size)
study_hours = np.random.randint(1, 8, size)
previous_score = np.random.randint(30, 100, size)

# Realistic rule + noise
at_risk = ((marks < 50) | (attendance < 60)).astype(int)

noise_idx = np.random.choice(size, 50)
at_risk[noise_idx] = 1 - at_risk[noise_idx]

df = pd.DataFrame({
    "marks": marks,
    "attendance": attendance,
    "study_hours": study_hours,
    "previous_score": previous_score,
    "at_risk": at_risk
})

# 2️⃣ Feature Engineering
df["effort_ratio"] = df["study_hours"] / df["attendance"]
df["performance_index"] = (df["marks"] + df["previous_score"]) / 2

# 3️⃣ Feature Selection (remove weak feature)
X = df[["marks","attendance","previous_score","effort_ratio","performance_index"]]
y = df["at_risk"]

# 4️⃣ Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

# 5️⃣ Models
models = {
    "Logistic": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ]),
    
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42))
    ]),
    
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC())
    ])
}

# 6️⃣ Train & Compare
best_model = None
best_score = 0

print("\nModel Comparison:\n")

for name, model in models.items():
    
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    score = model.score(X_test, y_test)
    
    print(f"{name} Accuracy: {score:.3f}")
    
    if score > best_score:
        best_score = score
        best_model = model
        best_name = name

# 7️⃣ Final Evaluation
print(f"\nBest Model: {best_name}")

final_pred = best_model.predict(X_test)

print("\nFinal Model Report:")
print(classification_report(y_test, final_pred))