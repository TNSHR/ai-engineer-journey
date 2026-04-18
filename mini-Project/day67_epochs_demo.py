import numpy as np

# Simple training simulation
epochs = 5

loss = 10

print("Training Start:\n")

for epoch in range(epochs):
    
    loss = loss * 0.7  # simulate improvement
    
    print(f"Epoch {epoch+1}, Loss: {loss}")