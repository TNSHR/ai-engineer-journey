import random

# Simple word prediction simulation
data = {
    "I love": ["AI", "coding", "Python"],
    "AI is": ["powerful", "amazing", "future"],
}

input_text = "I love"

print("Input:", input_text)

predictions = data.get(input_text, ["something"])

print("Predicted next word:", random.choice(predictions))