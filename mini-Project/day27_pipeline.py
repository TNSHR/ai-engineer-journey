import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = {
    "marks": [30,40,50,60,70,80,35,45,55,65],
    "attendance": [55,60,70,80,85,90,58,65,75,45],
    "result": [0,0,1,1,1,1,1,0,1,1]
}

df = pd.DataFrame(data)

X = df[["marks", "attendance"]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,random_state=42,stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
print(classification_report(y_test, predictions))