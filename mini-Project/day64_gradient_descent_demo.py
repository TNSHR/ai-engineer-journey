import numpy as np

# Simple example: minimize f(x) = x^2

def gradient(x):
    return 2 * x   # derivative of x^2

# Initial value
x = 10

learning_rate = 0.1

# Run gradient descent
for i in range(10):
    grad = gradient(x)
    x = x - learning_rate * grad
    
    print(f"Step {i+1}: x = {x}")