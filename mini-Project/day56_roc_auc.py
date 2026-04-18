import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# Dataset
np.random.seed(42)

size = 200

marks = np.random.randint(30, 100, size)
attendance = np.random.randint(50, 100, size)

at_risk = (marks < 50).astype(int)

df = pd.DataFrame({
    "marks": marks,
    "attendance": attendance,
    "at_risk": at_risk
})

X = df[["marks","attendance"]]
y = df["at_risk"]

# Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipeline.fit(X_train,y_train)

# 🔥 Get probabilities
y_probs = pipeline.predict_proba(X_test)[:,1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# AUC score
roc_auc = auc(fpr, tpr)

print("AUC Score:", roc_auc)