import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Data
data = {
    "marks":[35,50,65,80,40,55,70,30,85,60],
    "attendance": [60,70,85,90,65,75,88,55,92,80],
    "at_risk":[1,0,0,0,1,0,0,1,0,0]
}

df = pd.DataFrame(data)

X = df[["marks", "attendance"]]
y = df["at_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 42
)

#Different number of tree
trees = [1,5,50]

for n in trees:
    model = RandomForestClassifier(
        n_estimators = n,
        random_state = 42
    )

    model.fit(X_train,y_train)
    y_pred= model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\nNumber of Trees:", n)
    print("Accuracy", acc)