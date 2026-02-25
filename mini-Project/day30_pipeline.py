import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#1. Create a dataset (student performance)
data = {
    "marks": [35, 50, 65, 80, 40, 55, 70, 30, 85, 60],
    "attendance": [60, 70, 85, 90, 65, 75, 88, 55, 92, 80],
    "study_hours": [1, 2, 4, 6, 2, 3, 5, 1, 7, 4],
    "previous_score": [40, 55, 70, 85, 45, 60, 75, 35, 90, 68],
    "at_risk": [1, 0, 0, 0, 1, 1, 0, 1, 0, 0]
}

df = pd.DataFrame(data)
print("Original Data:")
print(df)
#2. Feature engineering
df["effort_ratio"] = df["study_hours"]/df["attendance"]
print("\nDataset after feature Engineering:")
print(df)

#3. Split  Features and Labels
X = df.drop("at_risk", axis=1)
y= df["at_risk"]

#4. Train Test Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
#5. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#6. Train Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaled,y_train)

#7.Prediction
y_pred = model.predict(X_test_scaled)

#8.Evaluation
print("\nClassification Report:")
print(classification_report(y_test,y_pred))

#9. show actual vs predicted
results_df = pd.DataFrame(X_test_scaled, columns = X.columns)
results_df["actual"] = y_test.values
results_df["predicted"] = y_pred

print("\nPrediction Comparison:")
print(results_df)
