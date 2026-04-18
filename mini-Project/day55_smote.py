import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE

#Create imbalanced dataset
np.random.seed(42)

size = 200000
marks = np.random.randint(30,100,size)
attendance = np.random.randint(50,100,size)

at_risk = np.zeros(size)
at_risk[:200] = 1
at_risk = (marks < 50) | (attendance < 60)

df = pd.DataFrame({
    "marks": marks,
    "attendance": attendance,
    "at_risk": at_risk
})

X = df[["marks", "attendance"]]
y = df["at_risk"]

#Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

#Without SMOTE
model1 = Pipeline([
    ("scaler",StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced", random_state=42))
])
model1.fit(X_train,y_train)
pred1 = model1.predict(X_test)  
print("\nWithout SMOTE:\n")
print(classification_report(y_test,pred1))

#With SMOTE
smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", pd.Series(y_train_resampled).value_counts())

#Train model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42))
])

pipeline.fit(X_train_resampled, y_train_resampled)
y_pred = pipeline.predict(X_test)
print("\nWith SMOTE:\n")
print(classification_report(y_test, y_pred))