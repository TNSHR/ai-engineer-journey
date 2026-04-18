import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1️⃣ Dataset
data = {
    "marks":[35,50,65,80,40,55,70,30,85,60],
    "attendance":[60,70,85,90,65,75,88,55,92,80],
    "study_hours":[1,2,4,6,2,3,5,1,7,4],
    "previous_score":[40,55,70,85,45,60,75,35,90,68],
    "at_risk":[1,0,0,0,1,0,0,1,0,0]
}

df = pd.DataFrame(data)

print("Original Data:\n", df)

# 2️⃣ Feature Engineering
df["effort_ratio"] = df["study_hours"] / df["attendance"]
df["performance_index"] = (df["marks"] + df["previous_score"]) / 2

print("\nAfter Feature Engineering:\n", df)

# 3️⃣ Features & Target
X = df.drop("at_risk", axis=1)
y = df["at_risk"]

# 4️⃣ Train-Test Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

# 5️⃣ Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=50, random_state=42))
])

# 6️⃣ Train
pipeline.fit(X_train,y_train)

# 7️⃣ Predict
y_pred = pipeline.predict(X_test)

# 8️⃣ Evaluation
print("\nModel Performance:")
print(classification_report(y_test,y_pred))

# 9️⃣ Feature Importance
model = pipeline.named_steps["model"]

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print("\nFeature Importance:\n", importance)