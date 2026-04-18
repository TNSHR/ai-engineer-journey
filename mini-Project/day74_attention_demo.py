sentence = ["I", "love", "AI"]

print("Attention Simulation:\n")

for word in sentence:
    print(f"\nWord: {word}")
    print("Attends to:")
    
    for other in sentence:
        print(f"  - {other}")