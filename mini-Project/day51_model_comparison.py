import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#Create largest dataset with noise
np.random.seed(42)
size = 1000

marks = np.random.randint(30,100,size)
attendance = np.random.randint(50,100,size)
study_hours = np.random.randint(1,10,size)
previous_score = np.random.randint(30,100,size)

#Rule-based target with noise
at_risk = (marks < 50).astype(int)

#Add noise
noise_idx = np.random.choice(size,30)
at_risk[noise_idx] = 1 - at_risk[noise_idx]

df = pd.DataFrame({
    "marks": marks,
    "attendance": attendance,
    "study_hours": study_hours,
    "previous_score": previous_score,
    "at_risk": at_risk
})

#Features Engineering
df["effort_ratio"] = df["study_hours"]/df["attendance"]
df["attendance"] = (df["marks"]+ df["previous_score"])/2

#Features & Target
X = df.drop("at_risk", axis=1)
y = df["at_risk"]

#split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

#Logistic Regression Pipeline
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42))
])
#Random Forest Pipeline
rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

#Train both models
lr_pipeline.fit(X_train,y_train)
rf_pipeline.fit(X_train,y_train)

#Predict
lr_pred = lr_pipeline.predict(X_test)
rf_pred = rf_pipeline.predict(X_test)

#Evaluate
print("\nLogistic Regression Performance:")
print(classification_report(y_test,lr_pred))
print("\nRandom Forest Performance:")
print(classification_report(y_test,rf_pred))
