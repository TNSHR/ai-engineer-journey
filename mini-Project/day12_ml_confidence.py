import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Simple dataset

data = {
    "marks": [20,35,50,65,80,90,45,55,70,85],
    "result" : [0,0,0,1,1,1,0,0,1,1]  # 1 for Pass, 0 for Fail
}

df = pd.DataFrame(data)
print("Dataset:", df)

X = df[["marks"]] #input features
y = df["result"] #output labels(What we want to predict)

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= 0.4, random_state=1
)
#Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

#Make predictions
predictions = model.predict(X_test)
print("Test Marks:\n", X_test.values.flatten())
print("Predicted Results (0=Fail, 1=Pass):\n", predictions)
print("Actual Results:\n", y_test.values)

#Calculate accuracy
accuracy = accuracy_score(y_test, predictions)  
print(f"Model Accuracy: {accuracy*100:.2f}%")