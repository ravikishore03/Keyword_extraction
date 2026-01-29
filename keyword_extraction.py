from sklearn.feature_extraction.text import TfidfVectorizer

#Input text (list of documents)
texts = ["Machine Learning imporves decision making.",
         "Deep Learning is a subset of Machine Learning.",
         "Artificial Intelligence encompasses Machine Learning and Deep Learning." ]

# Intialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the texts
tfidf_matrix = vectorizer.fit_transform(texts)

# Extract keywords (feature names)
keywords = vectorizer.get_feature_names_out()

# Print keywords
print("Extracted Keywords:", keywords)