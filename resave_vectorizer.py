import dill
from model.nlp_model import recipe_tokenizer  # ✅ Ensures scope is correct
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the original vectorizer
with open("model/tfidf_vectorizer (1).pkl", "rb") as f:
    vectorizer = dill.load(f)

# Re-save using correct scope
with open("model/tfidf_vectorizer (1).pkl", "wb") as f:
    dill.dump(vectorizer, f)

print("✅ Vectorizer re-saved with correct scope.")