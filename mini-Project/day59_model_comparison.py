import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

# Dataset
np.random.seed(42)

size = 300

marks = np.random.randint(30, 100, size)
attendance = np.random.randint(50, 100, size)
study_hours = np.random.randint(1, 8, size)
previous_score = np.random.randint(30, 100, size)

at_risk = (marks < 50).astype(int)

df = pd.DataFrame({
    "marks": marks,
    "attendance": attendance,
    "study_hours": study_hours,
    "previous_score": previous_score,
    "at_risk": at_risk
})

X = df.drop("at_risk", axis=1)
y = df["at_risk"]

# Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

# Models
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
    ]),
    
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ])
}

# Train & Evaluate
results = {}

for name, model in models.items():
    
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, pred)
    
    results[name] = acc

# Print results
print("\nModel Comparison Results:\n")

for name, acc in results.items():
    print(f"{name}: {acc:.3f}")