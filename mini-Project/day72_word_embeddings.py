from sklearn.feature_extraction.text import TfidfVectorizer

sentences = [
    "I love AI",
    "AI is amazing",
    "I love machine learning",
    "machine learning is powerful"
]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(sentences)

print("Vocabulary:")
print(vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix:")
print(X.toarray())