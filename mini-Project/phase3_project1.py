import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split ,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier


# data = {
#     "marks":[35,50,65,80,40,55,70,30,85,60],
#     "attendance": [60,70,85,90,65,75,88,55,92,80],
#     "study_hours":[1,2,4,6,2,3,5,1,7,4],
#     "previous_score":[40,55,70,85,45,60,75,35,90,68],
#     "at_risk":[1,0,0,0,1,0,0,1,0,0]
# }
# data = {
#     "marks": [35, 50, 65, 80, 40, 55, 70, 30, 85, 60],
#     "attendance": [60, 70, 85, 90, 65, 75, 88, 55, 92, 80],
#     "study_hours": [1, 2, 4, 6, 2, 3, 5, 1, 7, 4],
#     "previous_score": [40, 55, 70, 85, 45, 60, 75, 35, 90, 68],
#     # introduce realistic noise
#     "at_risk": [1, 0, 0, 0, 1, 1, 0, 1, 0, 0]
# }
np.random.seed(42)

n = 1000

marks = np.random.randint(30, 100, n)
attendance = np.random.randint(50, 100, n)
study_hours = np.random.randint(1, 8, n)
previous_score = np.random.randint(30, 100, n)

# base logic: weaker students more likely at risk
risk_score = (
    0.4 * (marks < 50).astype(int) +
    0.3 * (attendance < 70).astype(int) +
    0.2 * (study_hours < 3).astype(int) +
    0.3 * (previous_score < 50).astype(int)
)

# convert to probability
prob = risk_score / risk_score.max()

# introduce noise (real-world randomness)
noise = np.random.binomial(1, 0.1, n)

at_risk = ((prob + noise) > 0.5).astype(int)

df = pd.DataFrame({
    "marks": marks,
    "attendance": attendance,
    "study_hours": study_hours,
    "previous_score": previous_score,
    "at_risk": at_risk
})

# feature engineering
df["effort_ratio"] = df["study_hours"] / df["attendance"]

print(df.head())



#Features and target
X = df.drop("at_risk", axis=1)
y= df["at_risk"]

#Train-test split
X_train, X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42,stratify=y
)

#Scaling(fit only on training)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Baseline model
model = LogisticRegression()
model.fit(X_train,y_train)

#predictions
predictions = model.predict(X_test)
print(classification_report(y_test,predictions))

print(pd.DataFrame({
    "Feature":X.columns,
    "Coefficient":model.coef_[0]
}))

# Convert scaled test data back to DataFrame
X_test_df = pd.DataFrame(X_test, columns=X.columns)

result_df = X_test_df.copy()
result_df["Actual"] = y_test.values
result_df["Predicted"] = predictions

print(result_df)
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train,y_train)

tree_preds = tree_model.predict(X_test)

print("\nDecision Tree Classification Report:")
print(classification_report(y_test,tree_preds))

depths = [2,4,None]
for d in depths:
    tree = DecisionTreeClassifier(max_depth=d,random_state=42)
    tree.fit(X_train,y_train)
    preds = tree.predict(X_test)
    print(f"\nDecision Tree (max_depth={d}) Report:")
    print(classification_report(y_test,preds))


#Logistic Regression CV
log_cv = cross_val_score(model,X_train,y_train,cv=5,scoring="accuracy")
print("\nLogistic Regression CV Accuracy:",log_cv)
print("Mean CV Accuracy:",log_cv.mean())

#Decision Tree CV
tree_cv = cross_val_score(tree_model,X_train,y_train,cv=5,scoring="accuracy")
print("\nDecision Tree CV Accuracy:", tree_cv)
print("Mean CV Accurancy:", tree_cv.mean())









