import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#Dataset
data = {
    "marks":[35,50,65,80,40,55,70,30,85,60],
    "attendance":[60,70,85,90,65,75,88,55,92,80],
    "at_risk":[1,0,0,0,1,0,0,1,0,0]
}

df = pd.DataFrame(data)
X = df[["marks","attendance"]]
y = df["at_risk"]

#Pipeline
pipeline = Pipeline([("scaler",StandardScaler()),("knn",KNeighborsClassifier())])

#Hyperparameter grid
param_grid = {"knn__n_neighbors":[1,3,5]}

#GridSearch
grid = GridSearchCV(pipeline,param_grid,cv=3)

#Train
grid.fit(X,y)
print("Best Parameters:", grid.best_params_)
print("Best Score", grid.best_score_)