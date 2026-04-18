import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#Dataset
data = {
    "marks":[35,50,65,80,40,55,70,30,85,60],
    "attendance":[60,70,85,90,65,75,88,55,92,80],
    "at_risk":[1,0,0,0,1,0,0,1,0,0]
}

df = pd.DataFrame(data)

X = df[["marks","attendance"]]
y = df["at_risk"]

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.3, random_state=42
)
#Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Test different K values
k_values = [1,3,5,7]

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    print("\nK= ",k)
    print("Accuracy = ",acc)
