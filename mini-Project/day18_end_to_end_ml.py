import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#Load dataset

data = {
    "marks": [30,40,50,60,70,80,35,90,55,65],
    "study_hours":[1,2,3,4,5,6,1,7,3,4],
    "attendance":[55,65,70,80,85,90,60,95,75,82],
    "result":[0,1,1,1,1,1,0,1,1,1]
}

df = pd.DataFrame(data)

#Features and target variable
X = df[["marks", "study_hours", "attendance"]]
y = df["result"]

#Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

#probabilities and thresholding
probs = model.predict_proba(X_test)[:,1]
threshold = 0.6
predictions = (probs >= threshold).astype(int)

#Evaluation
print("Classification Report (Threshold = 0.6):", threshold)
print(classification_report(y_test, predictions))

#Explanation:
#Model outputs probabilities
#Threshold is adjusted to reduce false positives
#Evaluation uses predictions and recall, not only accuracy
#Data quality and feature matter more than model choice