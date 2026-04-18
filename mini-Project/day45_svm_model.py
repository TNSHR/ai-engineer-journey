import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

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
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#SVM model
model = SVC(kernel='linear')

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("SVM Classification Report:")
print(classification_report(y_test,y_pred))