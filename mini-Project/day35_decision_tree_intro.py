import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

#1 Create a simple dataset
data = {
    "marks": [35,50,65,80,90,55,70,30,85,60],
    "attendance": [60,70,85,90,65,75,88,55,92,80],
    "at_risk": [1,0,0,0,1,0,0,1,0,0]
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)

#2. Feature and target
X = df[["marks", "attendance"]]
y = df["at_risk"]

#3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#4 Train Decision Tree Classifier
tree = DecisionTreeClassifier(random_state = 42)
tree.fit(X_train,y_train)

#5. Predictions
y_pred = tree.predict(X_test)

#6.Evaluation
print("\nClassification Report:")
print(classification_report(y_test,y_pred))

#7. Feature importance
importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": tree.feature_importances_
})

print("\nFeature Importances:")
print(importances)