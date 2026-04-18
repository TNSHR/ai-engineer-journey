import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#1.Dataset Creation
data={
    "marks":[35,50,65,80,40,55,70,30,85,60],
    "attendance":[60,70,85,90,65,75,88,55,92,80],
    "study_hours":[1,2,4,6,2,3,5,1,7,4],
    "previous_score":[40,55,70,85,45,60,75,35,90,68],
    "at_risk":[1,0,0,0,1,1,0,1,0,0 ]
}
df = pd.DataFrame(data)
#2.Feature Engineering
df["effort_ratio"]=df["study_hours"]/df["attendance"]
print("Dataset:")
print(df)

#3.Feature and Target
X = df.drop("at_risk",axis=1)
y = df["at_risk"]

#4.Train Test Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#5.Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#6.Model Training
model = LogisticRegression()
model.fit(X_train_scaled,y_train)

#7.Prediction
y_pred = model.predict(X_test_scaled)

#8.Evaluation
print("\nClassification Report:")
print(classification_report(y_test,y_pred))

#9.Coefficients Analysis
coeff_df = pd.DataFrame(
    {
        "Feature":X.columns,
        "Coefficient": model.coef_[0]
    }
)

print("\nFeature Coefficients(model Decision Weights):")
print(coeff_df.sort_values(by="Coefficient", ascending=False))

#Decision Boundary Analusis
sample = X_test_scaled[0]
score = np.dot(sample,model.coef_[0]) + model.intercept_[0]
probability = 1/(1+np.exp(-score))

print("\nExample Analysis:")
print("Raw Score:", score)
print("Predicted Probability :",probability)
print("Final Prediction:", 1 if probability > 0.5 else 0)