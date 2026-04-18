import numpy as np

#Actual valus
y_true = np.array([1,0,1,1])
#Predicted probabilities
y_pred = np.array([0.9,0.2,0.8,0.4])

#MSE
mse = np.mean((y_true - y_pred)**2)
#Binary Cross-Entropy
epsilon = 1e-15
y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
bce = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)) 

print("MSE:", mse)
print("Binary Cross-Entropy:", bce)