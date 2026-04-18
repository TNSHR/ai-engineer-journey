import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Dataset
np.random.seed(42)

size = 300

marks = np.random.randint(30, 100, size)
attendance = np.random.randint(50, 100, size)

at_risk = (marks < 50).astype(int)

df = pd.DataFrame({
    "marks": marks,
    "attendance": attendance,
    "at_risk": at_risk
})

X = df[["marks","attendance"]]
y = df["at_risk"]

# Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

# Scaling (VERY IMPORTANT for NN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural Network
model = MLPClassifier(hidden_layer_sizes=(5,5), max_iter=500)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Neural Network Report:")
print(classification_report(y_test,y_pred))