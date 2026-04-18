import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#1 Clean Data
data_clean = {
    "marks": [35,50,65,80,90,55,70,30,85,60],
    "attendance": [60,70,85,90,65,75,88,55,92,80],
    "at_risk": [1,0,0,0,1,0,0,1,0,0]
}
df_clean = pd.DataFrame(data_clean)

#2 Add Noise (flip some labels intentionally)
df_noisy = df_clean.copy()
df_noisy.loc[2, "at_risk"] = 1 #incorrect Label
df_noisy.loc[7, "at_risk"] = 0 #incorrect Label

print("Clean Data:")
print(df_clean)

print("\nNoisy Data:")
print(df_noisy)

#Function to train and evaluate
def train_evaluate(df,name):
    X = df.drop("at_risk", axis=1)
    y = df["at_risk"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    model = LogisticRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test,y_pred))

#3.Train on clean data
train_evaluate(df_clean,"Clean Data")
#4.Train on Noisy data
train_evaluate(df_noisy,"Noisy Data")
