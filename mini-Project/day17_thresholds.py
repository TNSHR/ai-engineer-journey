import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

#Simple dataset

x = np.array([[30],[40],[50],[60],[70],[80],[90],[100]])
y = np.array([0,0,1,1,1,1,1,1])  # Binary target variable

#Train logistic regression model
model = LogisticRegression()
model.fit(x,y)

#Predict probabilities
probs = model.predict_proba(x)

print("Marks | Probability of Passing")
for mark, prob in zip(x.flatten(), probs[:,1]):
    print(mark, "   ->  ", round(prob, 4))

#Setting different thresholds
print("\nDecision with different thresholds=0.5")
for mark, prob in zip(x.flatten(), probs[:,1]):
    decision = "Pass" if prob >= 0.5 else "Fail"
    print(mark, " --> ", decision)
print("\nDecision with different thresholds=0.7")
for mark, prob in zip(x.flatten(), probs[:,1]):
    decision = "Pass" if prob >= 0.7 else "Fail"
    print(mark, " --> ", decision)