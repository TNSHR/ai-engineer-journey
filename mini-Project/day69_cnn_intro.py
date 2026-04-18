import numpy as np

# Simple "image" (5x5 matrix)
image = np.array([
    [1, 2, 3, 0, 1],
    [0, 1, 2, 3, 1],
    [1, 0, 1, 2, 2],
    [2, 1, 0, 1, 3],
    [1, 2, 1, 0, 1]
])

# Simple filter (edge detector)
filter = np.array([
    [1, 0],
    [0, -1]
])

# Convolution
output = []

for i in range(4):
    row = []
    for j in range(4):
        region = image[i:i+2, j:j+2]
        value = np.sum(region * filter)
        row.append(value)
    output.append(row)

output = np.array(output)

print("Original Image:\n", image)
print("\nFilter:\n", filter)
print("\nFeature Map:\n", output)