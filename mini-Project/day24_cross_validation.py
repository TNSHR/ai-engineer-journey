import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

data = {
    "marks": [40,50,60,70,80,35,45,55,65,45],
    "result": [0,0,1,1,1,1,0,1,1,1]

}


df = pd.DataFrame(data)
X= df[["marks"]]
y=df["result"]

model = LogisticRegression()

scores = cross_val_score(model,X,y,cv=5)
print("Cross-validation scores:", scores)
print("Mean cross-validation score:", scores.mean())