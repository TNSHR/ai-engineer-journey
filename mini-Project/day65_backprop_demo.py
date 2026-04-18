import numpy as np

#Input
x = 2
#Initial weight
w = 5
#Target output
y_true = 10
#Learning rate
lr = 0.1
#Forward pass
y_pred = w * x
#Loss (MSE)
loss = (y_true - y_pred)**2
#Gradient calculation (manual)

grad = -2 * x * (y_true - y_pred)
#Weight update
w = w - lr * grad
print(f"Updated weight: {w}")
print(f"Loss: {loss}")
print(f"Predicted value: {y_pred}")
print(f"True value: {y_true}")
print(f"Gradient: {grad}")