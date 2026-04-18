import numpy as np

epochs = 100

train_acc = []
test_acc = []

for i in range(epochs):
    train = 0.6 + i * 0.02 #keep incresing
    test = 0.6 + i * 0.015 #incresing then slows

    if i>50:
        test -= (i - 10) * 0.02 #overfitting
    
    train_acc.append(train)
    test_acc.append(test)


    print(f"Epoch {i+1} Train: {train:.2f}, Test: {test:.2f}")