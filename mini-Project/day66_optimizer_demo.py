import numpy as np

# Simple function: f(x) = x^2
def grad(x):
    return 2 * x

# Initial value
x = 10

# Learning rate
lr = 0.1

print("SGD Optimization:\n")

for i in range(10):
    x = x - lr * grad(x)
    print(f"Step {i+1}: x = {x}")