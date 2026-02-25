import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    "marks": [40,50,60,70,80,90,55,65,75,85],
    "passed": [0,0,1,1,1,1,0,1,1,1] 
}

df = pd.DataFrame(data)

X = df[["marks"]]
y = df["passed"]

#Fist, split the data into training and temp sets

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

#Second, split the temp set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Then, split the temp set into validation and test sets
model = LogisticRegression()
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

val_accuracy = accuracy_score(y_val, val_pred)
test_accuracy = accuracy_score(y_test, test_pred)
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

