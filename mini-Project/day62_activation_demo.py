import numpy as np

#Sample inputs
x = np.array([-10,-5,-1,0,1,5,10])

#Sigmoid 
def sigmoid(x):
    return 1/(1+np.exp(-x))

#ReLU
def relu(x):
    return np.maximum(0,x)
#Tanh
def tanh(x):
    return np.tanh(x)

print("Input:",x)
print("Sigmoid:",sigmoid(x))
print("ReLU:",relu(x))
print("Tanh:",tanh(x))