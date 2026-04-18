import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#Create imbalanced dataset
np.random.seed(42)

size = 200
marks = np.random.randint(30,100,size)
attendance = np.random.randint(50,100,size)

#Only few risky students
at_risk = np.zeros(size)
at_risk[:20] = 1 #only 10% at risky

df = pd.DataFrame({
    "marks":marks,
    "attendance": attendance,
    "at_risk": at_risk
})

X = df[["marks","attendance"]]
y = df["at_risk"]

#Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)
#Without class weight

model1 = Pipeline([
    ("scaler",StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced", random_state=42))
])

model1.fit(X_train,y_train)
pred1 = model1.predict(X_test)

print("\nWithout Class Weight:\n")
print(classification_report(y_test,pred1))

#With class weight

model2 = Pipeline([
    ("scaler",StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced", random_state=42))
])

model2.fit(X_train,y_train)
pred2 = model2.predict(X_test)

print("\nWith Class Weight:")
print(classification_report(y_test,pred2))