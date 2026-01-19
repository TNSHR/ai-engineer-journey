import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    "marks":[35,45,55,65,75,85,30,90],
    "study_hours":[1,2,3,4,5,6,1,7],
    "attendance":[60,65,70,80,85,90,55,95],
    "result":[0,1,1,1,1,1,0,1]
}

df = pd.DataFrame(data)
print(df)

X = df[["marks", "study_hours", "attendance"]]
y = df["result"]

X_train,X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Predictions:", predictions)
print("Actual:", y_test.values)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)