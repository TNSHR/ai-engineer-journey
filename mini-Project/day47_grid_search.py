import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#Dataset
data = {
    "marks":[35,50,65,80,40,55,70,30,85,60],
    "attendance":[60,70,85,90,65,75,88,55,92,80],
    "at_risk": [1,0,0,0,1,0,0,1,0,0]
}

df = pd.DataFrame(data)

X = df[["marks","attendance"]]
y = df["at_risk"]

#Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Model
model = KNeighborsClassifier()

#Hyperparamere grid
param_grid = {
    "n_neighbors":[1,3,5,7]
}

#Grid serach
grid = GridSearchCV(model,param_grid,cv=3)

grid.fit(X,y)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)