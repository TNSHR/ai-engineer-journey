import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

#1. Dataset
data = {
    "marks": [35,50,65,80,90,55,70,30,85,60],
    "attendance": [60,70,85,90,65,75,88,55,92,80],
    "at_risk": [1,0,0,0,1,0,0,1,0,0]
}
df = pd.DataFrame(data)

X = df[["marks", "attendance"]]
y = df["at_risk"]

#Model
model = DecisionTreeClassifier(max_depth=3,random_state = 42)

# 5-Fold cross Validation

scores = cross_val_score(model,X,y,cv=5)

print("Cross-Validation Scores:", scores)
print("Average CV Score:", scores.mean())