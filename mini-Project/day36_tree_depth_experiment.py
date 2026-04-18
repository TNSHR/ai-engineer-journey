import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#1 Create a simple dataset
data = {
    "marks": [35,50,65,80,90,55,70,30,85,60],
    "attendance": [60,70,85,90,65,75,88,55,92,80],
    "at_risk": [1,0,0,0,1,0,0,1,0,0]
}
df = pd.DataFrame(data)

X = df[["marks", "attendance"]]
y = df["at_risk"]

#2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#3. Different tree depths
depths = [1,3,None]

for d in depths:
    tree = DecisionTreeClassifier(max_depth = d,random_state = 42)
    tree.fit(X_train,y_train)
    y_pred = tree.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    print("\nTree Depth:", d)
    print("Accuracy:", acc)