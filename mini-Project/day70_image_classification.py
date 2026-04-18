import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report

# Load dataset
digits = load_digits()

X = digits.data
y = digits.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model (simulating CNN with MLP)
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Image Classification Report:\n")
print(classification_report(y_test, y_pred))