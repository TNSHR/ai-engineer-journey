sentence = ["I", "love", "AI"]

memory = ""

print("RNN Simulation:\n")

for word in sentence:
    memory += word + " "
    print(f"Current Word: {word}")
    print(f"Memory: {memory}\n")