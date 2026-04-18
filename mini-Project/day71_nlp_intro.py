from sklearn.feature_extraction.text import CountVectorizer

# Sample text
sentences = [
    "I love AI",
    "I love coding",
    "AI is powerful"
]

# Bag of Words
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(sentences)

print("Vocabulary:")
print(vectorizer.get_feature_names_out())

print("\nEncoded Matrix:")
print(X.toarray())