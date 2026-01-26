"""
Docstring for ai-engineer-journey.mini-Project.student_pass_prediction
Student Pass Prediction using Machine Learning
This project predicts whether a student will pass or fail based on marks, study hours, and attendance.
key focus:
  - End-to-end ML workflow
  -Threshold-based decision making
  - Proper model evaluation
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#Load dataset
data = {
    "marks": [30,45,50,60,70,80,35,90,55,65],
    "study_hours":[1,2,3,4,5,6,1,7,3,4],
    "attendance": [55,65,70,80,85,90,60,95,75,82],
    "result":[0,1,1,1,1,1,1,0,1,1]
}
df = pd.DataFrame(data)

#Features and target variable

X = df[["marks", "study_hours", "attendance"]]
y = df["result"]

#Train-test Split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42,stratify=y)

#Model Training

model = LogisticRegression()
model.fit(X_train, y_train)

#Threshold-based prediction

probs = model.predict_proba(X_test)[:,1]
predictions = (probs >= 0.6).astype(int)

#Model Evaluation

print("Threshold:", 0.6)
print(classification_report(y_test,predictions))

#Key insights:
#1. Model outputs probabilities for pass/fail
#2. Threshold adjustment helps balance false positives/negatives
#3. Evaluation focuses on recall and precision, not just accuracy
#4. Data quality and feature selection are crucial for model performance    
#5. Small datasets may lead to overfitting; larger datasets are preferable
#6. Better data would improve model result than a new model choice