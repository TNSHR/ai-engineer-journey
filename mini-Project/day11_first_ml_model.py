import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    "marks": [65,80,90,34,55,78,88,45,60,72],
    "result" : [1,1,1,0,1,1,1,0,1,1]  # 1 for Pass, 0 for Fail
}
df = pd.DataFrame(data)
print(df)

X = df[["marks"]] #input features
y = df["result"] #output labels(What we want to predict)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

#Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

#Make predictions
predictions = model.predict(X_test)
print("Predictions:", predictions)
print("Actual Labels:", y_test.values)

#Accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy*100:.2f}%")