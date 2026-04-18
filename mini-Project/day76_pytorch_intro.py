import torch

# Data
x = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y = torch.tensor([[2.0],[4.0],[6.0],[8.0]])

# Initialize weights
w = torch.tensor([[1.0]], requires_grad=True)
b = torch.tensor([[0.0]], requires_grad=True)

# Training loop
learning_rate = 0.01

for epoch in range(10):
    
    # Forward pass
    y_pred = x * w + b
    
    # Loss (MSE)
    loss = ((y - y_pred)**2).mean()
    
    # Backward (compute gradient)
    loss.backward()
    
    # Update weights
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    # Reset gradients
    w.grad.zero_()
    b.grad.zero_()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")