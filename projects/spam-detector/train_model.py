# train_model.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Sample data (text, label)
texts = [
    "Win a free iPhone now",        # spam
    "Limited offer just for you",   # spam
    "Hello, how are you?",          # ham
    "Let's have lunch tomorrow",    # ham
]
labels = [1, 1, 0, 0]  # 1 = spam, 0 = ham

# Create pipeline: vectorizer + classifier
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(texts, labels)

# Export the model
joblib.dump(model, 'spam_model.pkl')

print("Model trained and saved to spam_model.pkl")